package org.firstinspires.ftc.teamcode.drive;

import androidx.annotation.NonNull;

import com.acmerobotics.dashboard.config.Config;
import com.acmerobotics.roadrunner.geometry.Pose2d;
import com.acmerobotics.roadrunner.localization.TwoTrackingWheelLocalizer;
import com.qualcomm.robotcore.hardware.DcMotorEx;


import org.firstinspires.ftc.teamcode.util.Encoder;

import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleSupplier;

/*
 * Sample tracking wheel localizer implementation assuming the standard configuration:
 *
 *    ^
 *    |
 *    | ( x direction)
 *    |
 *    v
 *    <----( y direction )---->

 *        (forward)
 *    /--------------\
 *    |     ____     |
 *    |     ----     |    <- Perpendicular Wheel
 *    |           || |
 *    |           || |    <- Parallel Wheel
 *    |              |
 *    |              |
 *    \--------------/
 *
 */
@Config
public class TwoWheelTrackingLocalizer extends TwoTrackingWheelLocalizer {
    public static double TICKS_PER_REV = 8192;
    public static double TICKS_PER_GOBILDA = 8192; //2000

    public static double WHEEL_RADIUS = 0.62992; // in
    public static double WHEEL_GOBILDA = 0.944882;
    public static double GEAR_RATIO = 1; // output (wheel) speed / input (encoder) speed

    public static double PARALLEL_X = 0; // X is the up and down direction
    public static double PARALLEL_Y = 8; // Y is the strafe direction

    public static double PERPENDICULAR_X = -8;
    public static double PERPENDICULAR_Y = 0;

    public static double MULTIPLIER_X = -1.0 * 2/ 3; // 0.9911894273127753;
    public static double MULTIPLIER_Y = 1.0 * 2/  3; // 0.9929166666666667;

    // Parallel/Perpendicular to the forward axis
    // Parallel wheel is parallel to the forward axis
    // Perpendicular is perpendicular to the forward axis
    private Encoder parallelEncoder, perpendicularEncoder;
    private DoubleSupplier imuOrientation, imuVelocity;

    public TwoWheelTrackingLocalizer(DcMotorEx parallel, DcMotorEx perpendicular, DoubleSupplier imuOrientation, DoubleSupplier imuVelocity) {
        super(Arrays.asList(
                new Pose2d(PARALLEL_X, PARALLEL_Y, 0),
                new Pose2d(PERPENDICULAR_X, PERPENDICULAR_Y, Math.toRadians(90))
        ));

        parallelEncoder = new Encoder(parallel);
        perpendicularEncoder = new Encoder(perpendicular);

        this.imuOrientation = imuOrientation;
        this.imuVelocity = imuVelocity;

        // TODO: reverse any encoders using Encoder.setDirection(Encoder.Direction.REVERSE)
        parallelEncoder.setDirection(Encoder.Direction.FORWARD);
        perpendicularEncoder.setDirection(Encoder.Direction.REVERSE);
    }

    @Override
    public void setPoseEstimate(@NonNull Pose2d value) {
        super.setPoseEstimate(value);

    }

    public static double encoderTicksToInches(double ticks) {
        return WHEEL_RADIUS * 2 * Math.PI * GEAR_RATIO * ticks / TICKS_PER_REV;
    }

    public static double encoderTicksToInchesGobilda(double ticks) {
        return WHEEL_GOBILDA * 2 * Math.PI * GEAR_RATIO * ticks / TICKS_PER_GOBILDA;
    }

    @Override
    public double getHeading() {
        return imuOrientation.getAsDouble();
    }

    @Override
    public Double getHeadingVelocity() {
        return imuVelocity.getAsDouble();
    }

    @NonNull
    @Override
    public List<Double> getWheelPositions() {
        return Arrays.asList(
                encoderTicksToInchesGobilda(parallelEncoder.getCurrentPosition()) * MULTIPLIER_X,
                encoderTicksToInchesGobilda(perpendicularEncoder.getCurrentPosition()) * MULTIPLIER_Y
        );
    }

    @NonNull
    @Override
    public List<Double> getWheelVelocities() {
        return Arrays.asList(
                encoderTicksToInchesGobilda(parallelEncoder.getCorrectedVelocity()) * MULTIPLIER_X,
                encoderTicksToInchesGobilda(perpendicularEncoder.getCorrectedVelocity()) * MULTIPLIER_Y
        );
    }
}