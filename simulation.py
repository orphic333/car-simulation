from dataclasses import dataclass, field

import pygame

from car_class import Car, Drivetrain, Engine, torque_curve_fractions


@dataclass
class MotionRecorder:
    """Collect motion snapshots at a fixed interval."""

    interval_seconds: float = 0.2
    records: list[dict] = field(default_factory=list)
    _time_since_record: float = 0.0

    def __post_init__(self) -> None:
        if self.interval_seconds <= 0:
            raise ValueError("interval_seconds must be greater than zero")

    def update(self, car: Car, delta_time: float, elapsed_time: float) -> None:
        if delta_time < 0:
            raise ValueError("delta_time must not be negative")
        self._time_since_record += delta_time
        while self._time_since_record + 1e-12 >= self.interval_seconds:
            self.records.append(car.snapshot(elapsed_time))
            self._time_since_record -= self.interval_seconds


class KeyboardController:
    """Translate Pygame keyboard state into simulation input."""

    def acceleration_requested(self) -> bool:
        return bool(pygame.key.get_pressed()[pygame.K_UP])

    def braking_requested(self) -> bool:
        #ADDED: DOWN arrow = brake
        return bool(pygame.key.get_pressed()[pygame.K_DOWN])


def run(car: Car, wheel_radius: float, window_size: tuple[int, int] = (640, 240)) -> list[dict]:
    """Run the interactive acceleration simulation until the window closes."""
    pygame.init()
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Car motion")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 28)
    controller = KeyboardController()
    recorder = MotionRecorder()
    elapsed_time = 0.0
    car_width = 19  #previous value was 60px
    car_center_x = 40 + car_width // 2
    running = True

    try:
        while running:
            delta_time = clock.tick(60) / 1000.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_a:
                    car.abs_enabled = not car.abs_enabled   #ADDED (step two): A toggles ABS on/off

            elapsed_time += delta_time
            car.update_motion(
                acceleration_requested=controller.acceleration_requested(),
                braking_requested=controller.braking_requested(),
                delta_time=delta_time,
                wheel_radius=wheel_radius,
                upshift_rpm=7200,
            )
            recorder.update(car, delta_time, elapsed_time)

            screen.fill((30, 30, 30))
            ground_y = window_size[1] - 55
            pygame.draw.line(screen, (100, 100, 100), (0, ground_y + 6),  #previous value was (0, ground_y + 24)
                             (window_size[0], ground_y + 6), 2)         #previous value wsa (window_size[0], ground_y + 24)

            car_center_x += car.speed_mps * delta_time * 4
            if car_center_x > window_size[0] + car_width // 2:
                car_center_x = -car_width // 2
            elif car_center_x < -car_width // 2:
                car_center_x = window_size[0] + car_width // 2
            
            car_x = int(car_center_x - car_width // 2)
            pygame.draw.rect(screen, (200, 40, 40), (car_x, ground_y, car_width, 6)) #previous value ... car_width, 24))
            pygame.draw.circle(screen, (20, 20, 20), (car_x + 4, ground_y + 6), 2)   #previous values (car_x + 14, ground_y + 25, 7)
            pygame.draw.circle(screen, (20, 20, 20), (car_x + 15, ground_y + 6), 2)  #previous values (car_x + 48, ground_y + 25, 7)

            speed_text = font.render(
                f"Speed: {car.speed_mps * 2.237:6.1f} mph", True, (240, 240, 240)
            )
            rpm_text = font.render(
                f"RPM: {car.engine.rpm:6.0f}", True, (240, 240, 240)
            )
            gear_text = font.render(
                f"Gear: {car.drivetrain.current_gear}", True, (240, 240, 240)
            )
            front_load, rear_load = car.axle_loads()   #ADDED: live axle loads incl. weight transfer
            total_load = front_load + rear_load
            load_text = font.render(
                f"Load F/R: {front_load / total_load * 100:3.0f}% / {rear_load / total_load * 100:3.0f}%",
                True, (240, 240, 240)
            )
            abs_text = font.render(
                f"ABS: {'ON' if car.abs_enabled else 'OFF'}", True, (240, 240, 240)
            )
            brake_status_text = font.render(
                f"Brakes F/R: {car.front_brake_status} / {car.rear_brake_status}", True, (240, 240, 240)
            )
            brake_text = font.render(
                f"Brake: {car.brake_input * 100:3.0f}%", True, (240, 240, 240)
            )
            controls_text = font.render(
                "UP: accelerate | DOWN: brake | A: toggle ABS", True, (180, 180, 180)
            )
            screen.blit(speed_text, (20, 20))
            screen.blit(rpm_text, (20, 50))
            screen.blit(gear_text, (20, 80))
            screen.blit(brake_text, (20, 110))
            screen.blit(load_text, (300, 20))
            screen.blit(abs_text, (300, 50))
            screen.blit(brake_status_text, (300, 80))
            screen.blit(controls_text, (20, 145))
            pygame.display.flip()
    finally:
        pygame.quit()

    return recorder.records


if __name__ == "__main__":
    car = Car(
        model="Mustang GT",                        
        mass=1738,
        frontal_area=2.2,
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
    records = run(car, wheel_radius=0.3505)
    print(f"Recorded {len(records)} motion samples.")
