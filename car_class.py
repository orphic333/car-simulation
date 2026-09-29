import bisect
import math
import scipy.constants as const

# ---------------------------------------------------------------------------------------------
# 19/09/2026
# UNITS NOTE (changed): engine torque in this file is stored in lb·ft, but the force maths is SI
# (wheel_radius in metres, mass in kg, forces in newtons). Torque is now converted at the point of
# use, in Drivetrain.calculate_wheel_output:
#
#     torque [N·m] = torque [lb·ft] x 1.3558179483      (1 lb·ft = 4.4482216153 N x 0.3048 m)
#     torque [lb·ft] = torque [N·m] / 1.3558179483
#
# Before this change lb·ft was treated as N·m directly, so wheel force was ~26% too low.
# ---------------------------------------------------------------------------------------------
LB_FT_TO_NM = const.pound_force * const.foot

# ---------------------------------------------------------------------------------------------
# 20/09/2026
# BRAKING NOTE (added): braking is modelled as one more opposing force in Car.update_motion, next to
# rolling resistance and aero drag:
#
#     brake_force [N] = brake_input x BRAKE_FRICTION_COEFFICIENT x mass [kg] x g          (0 <= brake_input <= 1)
#     (simple total-grip form, step one. See the PER-AXLE BRAKING NOTE below for the current per-axle form.)
#     net_force       = drive_force - rolling_resistance - aero_drag - brake_force
#
# BRAKE_FRICTION_COEFFICIENT is peak braking deceleration expressed in g, and it doubles as the tire grip
# cap (brakes act on all four tires, so the full vehicle weight is used, not just the driven axle's share).
# Source: Car and Driver S650 Mustang GT test (via FastestLaps): 70-0 mph in 153 ft, 100-0 mph in 312 ft.
#     a = v^2 / (2d):   70 mph -> 31.29 m/s over 46.63 m = 10.50 m/s^2
#                       100 mph -> 44.70 m/s over 95.10 m = 10.51 m/s^2      10.5 / 9.80665 = 1.07 g
# Stopping distances already include drag and rolling resistance, so the brake-only figure is a few percent
# lower (about 1.045). 1.07 is the agreed starting value; retune here if you want an exact match.
# Tested car: Performance Pack brakes and Pirelli P Zero tires; a base-spec GT would likely stop longer.
# ---------------------------------------------------------------------------------------------
BRAKE_FRICTION_COEFFICIENT = 1.07

# ---------------------------------------------------------------------------------------------
# 20/09/2026
# PER-AXLE BRAKING NOTE (added, step two): brake force is now made by each axle separately.
#
#     total_demand [N]    = brake_input x BRAKE_SYSTEM_CAPACITY_G x mass x g       (what the brake hardware asks for)
#     front_demand        = total_demand x brake_bias_front            (braking force at front wheels)
#     rear_demand         = total_demand x (1 - brake_bias_front)       (braking force at rear wheels)
#     axle_grip_limit [N] = BRAKE_FRICTION_COEFFICIENT x that axle's load          (loads include weight transfer)
#     axle_force          = min(axle_demand, axle_grip_limit)                       (ABS on)
#     brake_force         = front_force + rear_force
#
# ABS on (default, Car.abs_enabled = True): an axle asking for more than its grip is held at its limit, so at
# full pedal both axles sit at their limits and the total is BRAKE_FRICTION_COEFFICIENT x mass x g, the same
# as step one, so the Car and Driver stopping distances above still hold. Bias only changes how much each
# axle brakes at lighter pedal.
# ABS off (Car.abs_enabled = False): an axle whose demand exceeds its grip LOCKS and only delivers
# LOCKED_GRIP_RATIO (0.8) of its limit, and stays locked until demand falls below that level. Total braking
# is then lower, and if the bias is too rearward the light rear axle locks first.
# ASSUMPTIONS (not published): brake_bias_front = 0.65, BRAKE_SYSTEM_CAPACITY_G = 1.5 (enough hardware to
# lock all four tires on dry pavement), LOCKED_GRIP_RATIO = 0.8 (sliding grip is roughly 70-85% of peak).
# Lockup is only reported (Car.front_brake_status / rear_brake_status: "ok", "limit" or "locked"); steering
# and spin-outs are not modelled yet.
# ---------------------------------------------------------------------------------------------
BRAKE_SYSTEM_CAPACITY_G = 1.5
LOCKED_GRIP_RATIO = 0.8

# ---------------------------------------------------------------------------------------------
# Date: 20/09/2026
# WEIGHT TRANSFER NOTE (added): when the car accelerates or brakes, weight shifts between the axles.
#
#     transfer [N]   = mass [kg] x longitudinal_acceleration [m/s^2] x cg_height [m] / wheelbase [m]
#     rear_load [N]  = mass x g x (1 - mass_distribution) + transfer
#     front_load [N] = mass x g x mass_distribution       - transfer      (mass_distribution = front share)
#
# longitudinal_acceleration is positive when speeding up (load moves to the rear) and negative when
# braking (load moves to the front). The total load always stays mass x g. Implemented in Car.axle_loads().
# The previous frame's acceleration is used, which avoids a circular dependency (grip depends on
# acceleration, which depends on grip) and costs only a one-frame lag.
# wheelbase = 2.72 m is the S650 figure (107.1 in). cg_height = 0.5 m is an ASSUMPTION (not published;
# typical for a sports coupe), so retune it. The drive traction limit uses the driven axle's load. The brake
# cap still uses total weight, which weight transfer doesn't change; per-axle braking is a later step.
# ---------------------------------------------------------------------------------------------

# (RPM, Fraction_of_Peak_Torque)
torque_curve_fractions = [
    (1500, 0.77),  # Strong low-end start, roughly ~320 lb-ft
    (2000, 0.82),  # Surges past the 80% mark early on
    (2500, 0.85),
    (3000, 0.87),  # Smooth, predictable climb
    (3500, 0.90),  # Enters the primary power band (90% of max torque available)
    (4000, 0.94),
    (4500, 0.98),  # Climbing toward the peak
    (4900, 1.00),  # PEAK TORQUE (100%). Corrected: 456.7 lb·ft = 619.2 N·m (with 480 hp @ 6000 rpm; was noted as 415-418 lb-ft).
                   # Not hard-coded: Engine.__init__ derives it from the hp/torque + rpm inputs
                   # (480 hp @ 6000 rpm / 0.92), so it changes if those inputs change.
    (5252, 0.98),  # The physical cross-over point with Horsepower
    (5500, 0.96),  # Torque begins its gradual mechanical taper
    (6000, 0.92),  # Falling off, but breathing well via dual throttle bodies
    (6500, 0.87),
    (7000, 0.81),  # Torque is down to ~81%, but Horsepower is near its absolute peak
    (7250, 0.76),  # Peak Horsepower RPM threshold
    (7500, 0.70)   # Approaching fuel cut-off / redline
]


class Engine:
    def __init__(self, engine_type:str, peak_horsepower=None, torque=None, volume=None, bore = None, stroke = None,
                 swept_volume = None, number_of_cylinders = None, rpm = None, redline_rpm = 7500,
                 torque_curve = None):
        #CHANGED: redline_rpm default raised from 6500 to 7500 so the car can use the full torque curve
        #(its last point is 7500 rpm). Note Car.update_motion's upshift_rpm default is still 5500.
        #User must fill in horsepower or torque, but not both.
        #Whichever is given is treated as the figure at `rpm`. If a torque_curve of
        #(rpm, fraction_of_peak_torque) points is supplied, that reference torque is used to
        #work out the engine's true peak torque. With no curve, torque stays constant as before.
        self.engine_type = engine_type
        self.peak_horsepower = peak_horsepower
        self.torque = torque
        self.bore = bore
        self.stroke = stroke
        self.swept_volume = swept_volume
        self.number_of_cylinders = number_of_cylinders
        self.rpm = rpm
        self.redline_rpm = redline_rpm

        if (peak_horsepower is None) == (torque is None):
            raise ValueError("Provide exactly one of peak_horsepower or torque")

        if rpm is None or rpm <= 0:
            raise ValueError("rpm must be greater than zero")
        if redline_rpm <= 0 or redline_rpm < rpm:
            raise ValueError("redline_rpm must be positive and at least the initial rpm")

        if peak_horsepower is None:
            self.peak_horsepower = (torque * rpm) / 5252
        else:
            self.torque = (peak_horsepower * 5252) / rpm

        if volume is None:
            if any(value is None for value in (self.bore, self.stroke, self.number_of_cylinders)):
                raise ValueError("bore, stroke, and number_of_cylinders are required when volume is not provided")
            bore = self.bore
            stroke = self.stroke
            number_of_cylinders = self.number_of_cylinders
            assert bore is not None and stroke is not None and number_of_cylinders is not None
            self.volume = round(math.pi * (bore / 2) ** 2 * stroke * number_of_cylinders, 4)
        else:
            self.volume = volume

        #Build the torque lookup table once here, so each frame only needs a single list index.
        #self.torque above is the torque at the reference rpm; dividing by the curve's fraction
        #at that rpm gives the true peak torque.
        fraction_table = self._build_fraction_table(torque_curve, self.redline_rpm)
        self.peak_torque = self.torque / fraction_table[int(rpm)]
        self._torque_table = [self.peak_torque * fraction for fraction in fraction_table]
        assert self.rpm is not None
        self.instantaneous_torque = self.torque_at(self.rpm)

    @staticmethod
    def _build_fraction_table(curve, max_rpm) -> list:
        """Linearly interpolate the curve into one fraction per whole rpm (0..max). Runs once."""
        points = sorted(curve) if curve else [(0, 1.0)]
        rpms = [point[0] for point in points]
        fractions = [point[1] for point in points]
        if any(fraction <= 0 for fraction in fractions):
            raise ValueError("torque_curve fractions must be greater than zero")
        if any(b == a for a, b in zip(rpms, rpms[1:])):
            raise ValueError("torque_curve must not repeat an rpm value")

        table = []
        for r in range(int(max(rpms[-1], max_rpm)) + 1):
            if r <= rpms[0]:
                table.append(fractions[0])      #below the first point: hold the first value
            elif r >= rpms[-1]:
                table.append(fractions[-1])     #above the last point: hold the last value
            else:
                i = bisect.bisect_right(rpms, r)
                r0, r1 = rpms[i - 1], rpms[i]
                f0, f1 = fractions[i - 1], fractions[i]
                table.append(f0 + (f1 - f0) * (r - r0) / (r1 - r0))
        return table

    def torque_at(self, rpm: float) -> float:
        """Return engine torque (lb-ft) at the given rpm from the precomputed table."""
        index = min(max(int(rpm + 0.5), 0), len(self._torque_table) - 1)
        return self._torque_table[index]

    def update_rpm(self, rpm: float) -> None:
        """Update engine speed (kept above idle) and refresh the torque that goes with it."""
        if rpm < 0:
            raise ValueError("rpm must not be negative")
        self.rpm = min(self.redline_rpm, max(800, int(rpm)))
        self.instantaneous_torque = self.torque_at(self.rpm)



class Drivetrain:
    def __init__(self, gears:list, final_drive:float, drivetrain_loss:float, mass_distribution:float,
                 drive_type:str, transmission_type:str):
        self.gears = gears
        self.final_drive = final_drive
        self.gear_index = 0
        self.drivetrain_loss = drivetrain_loss
        self.mass_distribution = mass_distribution
        self.drive_type = drive_type.upper()
        self.transmission_type = transmission_type

    @property
    def current_gear(self) -> int:
        """Return the active gear using one-based display numbering."""
        return self.gear_index + 1

    @property
    def current_gear_ratio(self) -> float:
       """Returns the current gears' ratio."""

       if 0 <= self.gear_index <len(self.gears):
           return self.gears[self.gear_index]
       return 0.0


    @property
    def total_multiplication(self) -> float:
        """Calculates the combined ratio of the current gear and final drive."""
        return self.current_gear_ratio * self.final_drive

    def maximum_speed_mps(self, wheel_radius: float, redline_rpm: float) -> float:
        """Return the road speed at which the current gear reaches redline."""
        if wheel_radius <= 0 or self.total_multiplication <= 0:
            raise ValueError("wheel_radius and total multiplication must be greater than zero")
        wheel_rpm = redline_rpm / self.total_multiplication
        return wheel_rpm * 2 * math.pi * wheel_radius / 60

    def calculate_traction_limit(self, vehicle_mass:float, friction_coefficient:float = 1.0,
                                 axle_loads: tuple | None = None) -> float:
        """Calculates the peak force (in Newtons) the drive tires can handle before slipping"""
        #ADDED: if the car supplies live axle loads (front_N, rear_N) that include weight transfer, use the
        #driven axle's load. Without them, fall back to the static split below (the old behaviour).
        if axle_loads is not None:
            front_load, rear_load = axle_loads
            if self.drive_type == "RWD":
                driven_load = rear_load
            elif self.drive_type == "FWD":
                driven_load = front_load
            else:
                driven_load = front_load + rear_load
            return driven_load * friction_coefficient

        if self.drive_type == "RWD":
            driven_weight_ratio = 1 - self.mass_distribution
            #FWD
        elif self.drive_type == "FWD":
            driven_weight_ratio = self.mass_distribution
            #AWD
        else:
            driven_weight_ratio = 1.0

        normal_force = vehicle_mass * driven_weight_ratio * const.g
        return normal_force * friction_coefficient

    def shift_up(self):
        """Shifts the gears to the next higher gear if available."""
        if self.gear_index < len(self.gears) - 1:
            self.gear_index = self.gear_index + 1

    def shift_down(self):
        """Shifts the gears to the previous lower gear if available."""
        if self.gear_index > 0:
            self.gear_index = self.gear_index - 1

    def update_transmission_logic(self, current_rpm:float, upshift_rpm:float, downshift_rpm:float, idle_rpm:float):
        """Automatically shifts gears based on the current RPM and defined thresholds."""
        #Only executes if the transmission is automatic
        if self.transmission_type.lower() != "automatic":
            return
        #Handle upshift logic
        #Check if we are revving past the upshift threshold and if we are not already in the highest gear
        if current_rpm > upshift_rpm and self.gear_index < len(self.gears) -1:
            self.shift_up()
        #Handle downshift logic
        else:
            #Determine the absolute minimum RPM allowed based on current gear
            if self.gear_index==0:
                min_allowed_rpm = idle_rpm
            else:
                min_allowed_rpm = downshift_rpm

            #Trigger downshift if RPM falls below our floor, ensuring we aren't already in 1st gear
            if current_rpm < min_allowed_rpm and self.gear_index > 0:
                self.shift_down()

    def calculate_wheel_output(self, torque: float, rpm: float, wheel_radius: float,
                               vehicle_mass: float, vehicle_forward_speed_mph: float,
                               friction_coefficient: float = 1.0, axle_loads: tuple | None = None) -> dict:
        """Calculate wheel output from the engine values for this simulation step. torque is in lb·ft.
        axle_loads is an optional (front_N, rear_N) pair; see calculate_traction_limit."""
        total_multiplication = self.total_multiplication
        if total_multiplication == 0 or wheel_radius <= 0:
            return {
                "actual_forward_force_newtons": 0.0,
                "wheel_slip_ratio": 0.0,
                "actual_wheel_rpm": 0.0,
                "is_spinning_out": False,
            }

        #Standard mechanical limits
        if torque < 0 or rpm < 0:
            raise ValueError("torque and rpm must not be negative")

        #CHANGED: convert engine torque (lb·ft) to N·m before dividing by wheel_radius (metres) to get newtons.
        #N·m = lb·ft x 1.3558179483 (see LB_FT_TO_NM at the top of this file).
        torque_nm = torque * LB_FT_TO_NM
        potential_wheel_torque = torque_nm * total_multiplication * (1 - self.drivetrain_loss)
        ideal_wheel_rpm = rpm / total_multiplication
        ideal_vehicle_forward_speed_mph = (ideal_wheel_rpm * 2 * math.pi * wheel_radius * 2.237)/60
        potential_tractive_effort = potential_wheel_torque / wheel_radius if wheel_radius > 0 else 0

        #Physical grip limits
        max_grip = self.calculate_traction_limit(vehicle_mass, friction_coefficient, axle_loads)

        #Wheel slip logic
        wheel_slip = 0.0
        actual_wheel_rpm = 0.0

        if potential_tractive_effort > max_grip:
            #Tires spin out. Forward force is capped at maximum tire grip capacity
            actual_forward_force = max_grip

            #Calculating wheel slip ratio
            #Prevent division by zero error if vehicle is perfectly stationary during a burnout
            speed_denominator = max(vehicle_forward_speed_mph,0.1)
            wheel_slip = (ideal_vehicle_forward_speed_mph - vehicle_forward_speed_mph)/speed_denominator #Reassignment

            #In a heavy spin, wheels rotate at full engine capability
            actual_wheel_rpm = ideal_wheel_rpm #Reassignment
        else:
            #Tires grip perfectly. Forward force matches engine output
            actual_forward_force = potential_tractive_effort

        return {
            'actual_forward_force_newtons': round(actual_forward_force, 3),
            'wheel_slip_ratio': max(0.0, wheel_slip), #Clamped to 0 minimum
            'actual_wheel_rpm': round(actual_wheel_rpm, 1),
            'is_spinning_out': potential_tractive_effort > max_grip,
        }

class Car:
    def __init__(self,
          model, mass, frontal_area, drivetrain:Drivetrain,
          car_class, car_id, engine:Engine, body_type,
          color="white", manufacturer = "Ford", wheelbase: float = 2.72, cg_height: float = 0.5,
          brake_bias_front: float = 0.65, abs_enabled: bool = True):
        if wheelbase <= 0 or cg_height < 0:
            raise ValueError("wheelbase must be greater than zero and cg_height must not be negative")
        if not 0.0 <= brake_bias_front <= 1.0:
            raise ValueError("brake_bias_front must be between 0 and 1")
        self.mass = mass
        self.frontal_area = frontal_area
        self.model = model
        self.drivetrain = drivetrain
        self.car_class = car_class
        self.car_id = car_id
        self.engine = engine
        self.body_type = body_type
        self.color = color
        self.manufacturer = manufacturer
        self.speed_mps = 0.0
        self.distance_m = 0.0
        self.brake_input = 0.0   #ADDED: brake pedal position, 0.0 (off) to 1.0 (full); ramps in update_motion
        self.wheelbase = wheelbase          #ADDED: metres (S650: 2.72)
        self.cg_height = cg_height          #ADDED: metres, centre of gravity height (assumed 0.5)
        self.longitudinal_acceleration = 0.0  #ADDED: m/s^2 from the last step, + speeding up / - braking
        self.brake_bias_front = brake_bias_front  #ADDED (step two): share of brake demand sent to the front axle
        self.abs_enabled = abs_enabled            #ADDED (step two): True = ABS holds tires at peak grip, False = axles can lock
        self.front_brake_status = "ok"            #ADDED (step two): "ok", "limit" (ABS holding) or "locked"
        self.rear_brake_status = "ok"

    def update_motion(
        self,
        acceleration_requested: bool,
        delta_time: float,
        wheel_radius: float,
        rolling_resistance_coefficient: float = 0.015,
        upshift_rpm: float = 5500.0,
        downshift_rpm: float = 1800.0,
        idle_rpm: float = 800.0,
        aerodynamic_drag_coefficient: float = 0.35,
        braking_requested: bool = False,
        brake_friction_coefficient: float = BRAKE_FRICTION_COEFFICIENT,
        brake_ramp_time: float = 0.15,   #CHANGED (step two): was 0.1; 0.15 s keeps the 70-0 mph stop near the 153 ft test figure
        brake_system_capacity_g: float = BRAKE_SYSTEM_CAPACITY_G,
    ) -> None:
        """Advance vehicle motion by one simulation step."""
        if delta_time <= 0:
            raise ValueError("delta_time must be greater than zero")
        if wheel_radius <= 0:
            raise ValueError("wheel_radius must be greater than zero")
        if rolling_resistance_coefficient < 0:
            raise ValueError("rolling_resistance_coefficient must not be negative")
        if aerodynamic_drag_coefficient < 0:
            raise ValueError("aerodynamic_drag_coefficient must not be negative")
        if brake_friction_coefficient < 0:
            raise ValueError("brake_friction_coefficient must not be negative")
        if brake_ramp_time < 0:
            raise ValueError("brake_ramp_time must not be negative")
        if brake_system_capacity_g < 0:
            raise ValueError("brake_system_capacity_g must not be negative")
        if self.engine.instantaneous_torque is None or self.engine.rpm is None:
            raise RuntimeError("engine must provide both torque and rpm")

        #ADDED: brake pedal. brake_input moves toward 1.0 (pressed) or 0.0 (released) by at most
        #delta_time / brake_ramp_time per step, so full force takes brake_ramp_time seconds (0 = instant).
        brake_target = 1.0 if braking_requested else 0.0
        max_step = 1.0 if brake_ramp_time == 0 else delta_time / brake_ramp_time
        self.brake_input += max(-max_step, min(max_step, brake_target - self.brake_input))

        rolling_resistance = (
            rolling_resistance_coefficient * self.mass * const.g
            if self.speed_mps > 0
            else 0.0
        )
        aerodynamic_resistance = 0.5 * 1.225 * aerodynamic_drag_coefficient * self.frontal_area * (self.speed_mps ** 2) #EDITED (29/09/2026): matches real world drag formula
        #CHANGED (step two): brake force is built per axle (see PER-AXLE BRAKING NOTE at the top of this file).
        #Zero when stopped. Axle loads include weight transfer, so the light rear axle reaches its limit first.
        brake_force = 0.0
        if self.speed_mps > 0 and self.brake_input > 0:
            front_load, rear_load = self.axle_loads()
            total_demand = self.brake_input * brake_system_capacity_g * self.mass * const.g
            front_force, self.front_brake_status = self._axle_brake_force(
                total_demand * self.brake_bias_front,
                brake_friction_coefficient * front_load,
                self.front_brake_status,
            )
            rear_force, self.rear_brake_status = self._axle_brake_force(
                total_demand * (1 - self.brake_bias_front),
                brake_friction_coefficient * rear_load,
                self.rear_brake_status,
            )
            brake_force = front_force + rear_force
        else:
            self.front_brake_status = self.rear_brake_status = "ok"
        drive_force = 0.0
        #CHANGED: any brake application overrides the throttle (drive force is cut while brake_input > 0).
        if acceleration_requested and self.brake_input == 0.0:
            output = self.calculate_wheel_output(
                wheel_radius=wheel_radius,
                vehicle_forward_speed_mph=self.speed_mps * 2.237,
            )
            drive_force = output["actual_forward_force_newtons"]

        wheel_rpm = (self.speed_mps / (2 * math.pi * wheel_radius)) * 60
        engine_rpm = wheel_rpm * self.drivetrain.total_multiplication
        self.engine.update_rpm(engine_rpm)
        self.drivetrain.update_transmission_logic(
            current_rpm=self.engine.rpm,
            upshift_rpm=upshift_rpm,
            downshift_rpm=downshift_rpm,
            idle_rpm=idle_rpm,
        )

        at_final_gear = self.drivetrain.gear_index == len(self.drivetrain.gears) - 1
        at_redline = self.engine.rpm >= self.engine.redline_rpm
        if at_final_gear and at_redline:
            # The rev limiter cuts drive force rather than allowing RPM to exceed redline.
            drive_force = 0.0
        # ADDED (29/09/2026): Electronic Speed Governor
        electronic_speed_limit_mph = 155.0
        limit_mps = electronic_speed_limit_mph / 2.237

        if self.speed_mps >= limit_mps:
            # The ECU cuts fuel/throttle, dropping drive force to zero
            drive_force = 0.0
        #CHANGED: brake_force added as another opposing term. Speed is clamped at 0 below, so braking
        #can never push the car backwards.
        net_force = drive_force - rolling_resistance - aerodynamic_resistance - brake_force
        acceleration = net_force / self.mass
        speed_before = self.speed_mps   #ADDED: used to record this step's acceleration for weight transfer
        self.speed_mps = max(0.0, self.speed_mps + acceleration * delta_time)
        self.distance_m += self.speed_mps * delta_time

        maximum_speed = self.drivetrain.maximum_speed_mps(
            wheel_radius, self.engine.redline_rpm
        )
        if at_final_gear:
            self.speed_mps = min(self.speed_mps, maximum_speed)
        wheel_rpm = (self.speed_mps / (2 * math.pi * wheel_radius)) * 60
        self.engine.update_rpm(wheel_rpm * self.drivetrain.total_multiplication)
        #ADDED: realised acceleration this step (after all clamps); drives weight transfer on the next step.
        self.longitudinal_acceleration = (self.speed_mps - speed_before) / delta_time

    def axle_loads(self) -> tuple:
        """ADDED: return (front_N, rear_N) tire loads including weight transfer (see WEIGHT TRANSFER NOTE)."""
        total_weight = self.mass * const.g
        static_front = total_weight * self.drivetrain.mass_distribution
        static_rear = total_weight - static_front
        transfer = self.mass * self.longitudinal_acceleration * self.cg_height / self.wheelbase
        #An axle can't carry negative load (it would be off the ground), so limit the transfer. Total stays mass x g.
        transfer = max(-static_rear, min(static_front, transfer))
        return static_front - transfer, static_rear + transfer

    def _axle_brake_force(self, demand: float, grip_limit: float, status: str) -> tuple:
        """ADDED (step two): return (force_N, status) for one axle. See PER-AXLE BRAKING NOTE."""
        if self.abs_enabled:
            #ABS trims brake pressure so the tire is held at its peak grip instead of locking.
            return (grip_limit, "limit") if demand > grip_limit else (demand, "ok")
        #ABS off: lock when demand exceeds grip; stay locked until demand drops below the sliding-grip level.
        locked = status == "locked"
        if locked and demand < LOCKED_GRIP_RATIO * grip_limit:
            locked = False
        elif not locked and demand > grip_limit:
            locked = True
        if locked:
            return LOCKED_GRIP_RATIO * grip_limit, "locked"
        return demand, "ok"

    def snapshot(self, elapsed_time: float) -> dict:
        """Return the values needed for a time-series motion record."""
        return {
            "time_s": round(elapsed_time, 3),
            "speed_mph": round(self.speed_mps * 2.237, 3),
            "engine_rpm": round(self.engine.rpm, 1), # type: ignore
            "gear": self.drivetrain.current_gear,
            "distance_m": round(self.distance_m, 3),
        }

    def calculate_wheel_output(self, wheel_radius: float, vehicle_forward_speed_mph: float,
                               friction_coefficient: float = 1.0) -> dict:
        """Coordinate the engine and drivetrain without coupling either class to the other."""
        if self.engine.instantaneous_torque is None or self.engine.rpm is None:
            raise RuntimeError("engine must provide both torque and rpm")

        return self.drivetrain.calculate_wheel_output(
            torque=self.engine.instantaneous_torque,
            rpm=self.engine.rpm,
            wheel_radius=wheel_radius,
            vehicle_mass=self.mass,
            vehicle_forward_speed_mph=vehicle_forward_speed_mph,
            friction_coefficient=friction_coefficient,
            axle_loads=self.axle_loads(),   #ADDED: live loads with weight transfer
        )

if __name__ == "__main__":
    myCar = Car(
        model="Mustang GT",
        mass=1738,
        frontal_area=2.2, # in square meters (updated: 29/09/2026)
        drivetrain=Drivetrain(
            gears=[4.70, 2.99, 2.15, 1.77, 1.52, 1.28, 1.00, 0.85, 0.69, 0.64],   #10-speed automatic (10R80)
            final_drive=3.15,   #Standard value. Performance Package ups this to Torrent 3.55:1
            drivetrain_loss=0.15,
            mass_distribution=0.55,
            drive_type="RWD",
            transmission_type="automatic"
        ),
        car_class="Sports Car",
        car_id="001",
        engine=Engine(
            engine_type="V8",
            peak_horsepower=480,
            torque=None,
            bore=3.66,
            stroke=3.56,
            number_of_cylinders=8,
            rpm=6000,
            torque_curve=torque_curve_fractions
        ),
        body_type="Coupe",
        color="Red",
        manufacturer="Ford"
    )

    print(f"Car Model: {myCar.model}"
          f"\nEngine Type: {myCar.engine.engine_type}"
          f"\nPeak Horsepower: {myCar.engine.peak_horsepower} HP"
          f"\nPeak Torque: {myCar.engine.peak_torque:.1f} lb-ft"
          f"\nTorque at {myCar.engine.rpm:.0f} rpm: {myCar.engine.instantaneous_torque:.1f} lb-ft"
          f"\nEngine Volume: {myCar.engine.volume} cubic inches"
          f"\nDrivetrain Type: {myCar.drivetrain.drive_type}"
          f"\nTransmission Type: {myCar.drivetrain.transmission_type}"
          f"\nCurrent Gear Ratio: {myCar.drivetrain.current_gear_ratio}"
          f"\nTotal Multiplication: {myCar.drivetrain.total_multiplication}"
          f"\nWheel Output: {myCar.calculate_wheel_output(wheel_radius=0.3505, vehicle_forward_speed_mph=60)}")
