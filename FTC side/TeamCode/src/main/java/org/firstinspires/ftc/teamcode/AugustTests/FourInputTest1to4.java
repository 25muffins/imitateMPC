package org.firstinspires.ftc.teamcode.AugustTests;

import android.annotation.SuppressLint;
import android.content.res.AssetFileDescriptor;

import com.acmerobotics.dashboard.FtcDashboard;
import com.acmerobotics.dashboard.canvas.Canvas;
import com.acmerobotics.dashboard.config.Config;
import com.acmerobotics.dashboard.telemetry.MultipleTelemetry;
import com.acmerobotics.dashboard.telemetry.TelemetryPacket;
import com.acmerobotics.roadrunner.geometry.Pose2d;
import com.acmerobotics.roadrunner.geometry.Vector2d;
import com.acmerobotics.roadrunner.path.LineSegment;
import com.acmerobotics.roadrunner.path.Path;
import com.acmerobotics.roadrunner.path.PathSegment;
import com.qualcomm.hardware.rev.RevHubOrientationOnRobot;
import com.qualcomm.robotcore.eventloop.opmode.Autonomous;
import com.qualcomm.robotcore.eventloop.opmode.LinearOpMode;
import com.qualcomm.robotcore.hardware.DcMotor;
import com.qualcomm.robotcore.hardware.DcMotorEx;
import com.qualcomm.robotcore.hardware.DcMotorSimple;
import com.qualcomm.robotcore.hardware.IMU;
import com.qualcomm.robotcore.util.ElapsedTime;

import org.firstinspires.ftc.robotcore.external.Telemetry;
import org.firstinspires.ftc.robotcore.external.navigation.AngleUnit;
import org.firstinspires.ftc.teamcode.drive.TwoWheelTrackingLocalizer;
import org.firstinspires.ftc.teamcode.util.DashboardUtil;
import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import static org.firstinspires.ftc.teamcode.trajectorysequence.TrajectorySequenceRunner.COLOR_INACTIVE_TRAJECTORY;

@Config
@Autonomous(name = "FourInputTest1to4")
public class FourInputTest1to4 extends LinearOpMode {
    IMU imu;
    TwoWheelTrackingLocalizer localizer;
    private ElapsedTime timer = new ElapsedTime();
    private FtcDashboard dashboard;

    public static float[] goal1    = {10, -10, 0};  // x, y, heading
    public static float[] goal2    = {20, -20, 0};
    public static float[] startPos = {0, 0, 0};
    public static float GOAL_SWITCH_DIST = 2.0f;    // inches
    public static float VELOCITY_SCALE   = 1.0f;    // scale factor on output velocities
    public static String MODEL_FILE      = "Test1_Baseline.tflite";
    private static final float POS_DIV   = 72.0f;
    private static final float VEL_DIV   = 30.0f;
    private static final float ANG_DIV   = (float) Math.PI;

    private static final int HISTORY_STEPS    = 8;
    private static final int HISTORY_FEATURES = 9;
    private float[][][] historyBuf = new float[1][HISTORY_STEPS][HISTORY_FEATURES];
    private boolean historyInitialized = false;
    private float prevVx    = 0;
    private float prevVy    = 0;
    private float prevOmega = 0;

    @SuppressLint("DefaultLocale")
    @Override
    public void runOpMode() throws InterruptedException {

        DcMotorEx fl = hardwareMap.get(DcMotorEx.class, "fl");
        DcMotorEx bl = hardwareMap.get(DcMotorEx.class, "bl");
        DcMotorEx br = hardwareMap.get(DcMotorEx.class, "br");
        DcMotorEx fr = hardwareMap.get(DcMotorEx.class, "fr");

        fl.setMode(DcMotor.RunMode.RUN_WITHOUT_ENCODER);
        bl.setMode(DcMotor.RunMode.RUN_WITHOUT_ENCODER);
        br.setMode(DcMotor.RunMode.RUN_WITHOUT_ENCODER);
        fr.setMode(DcMotor.RunMode.RUN_WITHOUT_ENCODER);

        fl.setDirection(DcMotorSimple.Direction.REVERSE);
        bl.setDirection(DcMotorSimple.Direction.REVERSE);
        br.setDirection(DcMotorSimple.Direction.REVERSE);
        fr.setDirection(DcMotorSimple.Direction.FORWARD);

        Interpreter interpreter;
        try {
            interpreter = new Interpreter(loadModelFile(MODEL_FILE));
        } catch (IOException e) {
            throw new RuntimeException("Failed to load TFLite model: " + e.getMessage());
        }

        //print input details to telemetry for verification
        telemetry.addData("Model inputs", interpreter.getInputTensorCount());
        for (int i = 0; i < interpreter.getInputTensorCount(); i++) {
            telemetry.addData("  input " + i,
                    interpreter.getInputTensor(i).name() +
                            " shape=" + Arrays.toString(interpreter.getInputTensor(i).shape()));
        }
        telemetry.update();

        imu = hardwareMap.get(IMU.class, "imu");
        imu.initialize(new IMU.Parameters(new RevHubOrientationOnRobot(
                RevHubOrientationOnRobot.LogoFacingDirection.RIGHT,
                RevHubOrientationOnRobot.UsbFacingDirection.UP
        )));
        imu.resetYaw();
        localizer = new TwoWheelTrackingLocalizer(fl, bl,
                () -> AngleUnit.normalizeRadians(
                        imu.getRobotYawPitchRollAngles().getYaw(AngleUnit.RADIANS)),
                () -> imu.getRobotAngularVelocity(AngleUnit.RADIANS).zRotationRate);

        dashboard = FtcDashboard.getInstance();
        Telemetry telemetry = new MultipleTelemetry(this.telemetry, dashboard.getTelemetry());

        float goalSwitch = 0.0f;
        boolean done     = false;

        waitForStart();
        timer.reset();

        while (opModeIsActive() && !done) {

            localizer.update();
            Pose2d pose    = localizer.getPoseEstimate();
            float heading  = (float) getHeading();
            float sinTh    = (float) Math.sin(heading);
            float cosTh    = (float) Math.cos(heading);

            float[] convPose  = convertCoordinates((float) pose.getX(), (float) pose.getY());
            float[] convStart = convertCoordinates(startPos[0], startPos[1]);
            float[] convG1    = convertCoordinates(goal1[0], goal1[1]);
            float[] convG2    = convertCoordinates(goal2[0], goal2[1]);

            //relative to start
            float relX  = convPose[0] - convStart[0];
            float relY  = convPose[1] - convStart[1];
            float relG1X = convG1[0] - convStart[0];
            float relG1Y = convG1[1] - convStart[1];
            float relG2X = convG2[0] - convStart[0];
            float relG2Y = convG2[1] - convStart[1];

            float dist1 = (float) Math.sqrt(Math.pow(relG1X - relX, 2) + Math.pow(relG1Y - relY, 2));
            float dist2 = (float) Math.sqrt(Math.pow(relG2X - relX, 2) + Math.pow(relG2Y - relY, 2));

            if (dist1 <= GOAL_SWITCH_DIST && goalSwitch == 0.0f) {
                goalSwitch = 1.0f;
                telemetry.addData("gs", "switched to goal2");
            }
            if (dist2 <= GOAL_SWITCH_DIST && goalSwitch == 1.0f) {
                done = true;
            }

            //ideally use wheel odometry velocities if available
            //these are placeholders — replace with actual velocity source
            Pose2d vel = localizer.getPoseVelocity();
            float vx = 0;
            float vy = 0;
            float omega = 0;
            if (vel != null) {
                vx = (float) vel.getX();
                vy = (float) vel.getY();
                omega = (float) vel.getHeading();
            }

            //query (1, 8) [x/72, y/72, sin_th, cos_th, vx/30, vy/30, omega/pi, gs]
            float[][] query = new float[1][8];
            query[0][0] = relX / POS_DIV;
            query[0][1] = relY / POS_DIV;
            query[0][2] = sinTh;
            query[0][3] = cosTh;
            query[0][4] = vx / VEL_DIV;
            query[0][5] = vy / VEL_DIV;
            query[0][6] = omega / ANG_DIV;
            query[0][7] = goalSwitch;

            //waypoints (1, 2, 4) [x/72, y/72, sin_heading, cos_heading] per goal
            float[][][] waypoints = new float[1][2][4];
            waypoints[0][0][0] = relG1X / POS_DIV;
            waypoints[0][0][1] = relG1Y / POS_DIV;
            waypoints[0][0][2] = (float) Math.sin(goal1[2]);
            waypoints[0][0][3] = (float) Math.cos(goal1[2]);
            waypoints[0][1][0] = relG2X / POS_DIV;
            waypoints[0][1][1] = relG2Y / POS_DIV;
            waypoints[0][1][2] = (float) Math.sin(goal2[2]);
            waypoints[0][1][3] = (float) Math.cos(goal2[2]);

            //goalswitch: (1, 1)
            float[][] gs = new float[1][1];
            gs[0][0] = goalSwitch;

            //history per step: [x/72, y/72, theta/pi, sin_th, cos_th, vx/30, vy/30, omega/pi, gs]
            float[] newHistStep = new float[HISTORY_FEATURES];
            newHistStep[0] = relX / POS_DIV;
            newHistStep[1] = relY / POS_DIV;
            newHistStep[2] = heading / ANG_DIV;
            newHistStep[3] = sinTh;
            newHistStep[4] = cosTh;
            newHistStep[5] = vx / VEL_DIV;
            newHistStep[6] = vy / VEL_DIV;
            newHistStep[7] = omega / ANG_DIV;
            newHistStep[8] = goalSwitch;
            updateHistory(newHistStep);

            //output (1, 3) [vx_norm, vy_norm, omega_norm]
            float[][] output = new float[1][3];

            //inputs must match TFLite index order
            //verify order from telemetry printout at init
            Object[] inputs = new Object[4];
            Map<Integer, Object> outputs = new HashMap<>();
            outputs.put(0, output);

            //set by shape to handle any reordering
            for (int i = 0; i < interpreter.getInputTensorCount(); i++) {
                int[] shape = interpreter.getInputTensor(i).shape();
                if (Arrays.equals(shape, new int[]{1, 8})) {
                    inputs[i] = query;
                } else if (Arrays.equals(shape, new int[]{1, 2, 4})) {
                    inputs[i] = waypoints;
                } else if (Arrays.equals(shape, new int[]{1, HISTORY_STEPS, HISTORY_FEATURES})) {
                    inputs[i] = weightedHistoryBuf;
                } else if (Arrays.equals(shape, new int[]{1, 1})) {
                    inputs[i] = gs;
                }
            }

            double startTime = timer.milliseconds();
            interpreter.runForMultipleInputsOutputs(inputs, outputs);
            double inferenceTime = timer.milliseconds() - startTime;

            //denormalize
            float outVx = output[0][0] * VEL_DIV;
            float outVy = output[0][1] * VEL_DIV;
            float outOmega = output[0][2] * ANG_DIV;

            //store in case
            prevVx = outVx;
            prevVy = outVy;
            prevOmega = outOmega;


            //mecanum velocity
            float flPower = VELOCITY_SCALE * (outVx - outVy - outOmega);
            float blPower = VELOCITY_SCALE * (outVx + outVy - outOmega);
            float brPower = VELOCITY_SCALE * (outVx - outVy + outOmega);
            float frPower = VELOCITY_SCALE * (outVx + outVy + outOmega);

            //normalize if any power exceeds 1.0
            float maxPower = 1;
            fl.setPower(flPower / maxPower);
            bl.setPower(blPower / maxPower);
            br.setPower(brPower / maxPower);
            fr.setPower(frPower / maxPower);

            telemetry.addData("max power", Math.max(1.0f,
                    Math.max(Math.abs(flPower),
                            Math.max(Math.abs(blPower),
                                    Math.max(Math.abs(brPower), Math.abs(frPower))))));

            telemetry.addData("inferenceTime (ms)", inferenceTime);
            telemetry.addData("goalSwitch",          goalSwitch);
            telemetry.addData("dist1",               dist1);
            telemetry.addData("dist2",               dist2);
            telemetry.addData("pose (rel)",
                    String.format("x=%.2f y=%.2f h=%.2f", relX, relY, heading));
            telemetry.addData("output (norm)",
                    String.format("vx=%.3f vy=%.3f om=%.3f",
                            output[0][0], output[0][1], output[0][2]));
            telemetry.addData("output (real)",
                    String.format("vx=%.3f vy=%.3f om=%.3f", outVx, outVy, outOmega));
            telemetry.addData("powers",
                    String.format("fl=%.2f bl=%.2f br=%.2f fr=%.2f",
                            flPower/maxPower, blPower/maxPower,
                            brPower/maxPower, frPower/maxPower));
            telemetry.update();

            drawField(pose, relX, relY, relG1X, relG1Y, relG2X, relG2Y);
        }
    }
    private float[][][] rawHistoryBuf     = new float[1][HISTORY_STEPS][HISTORY_FEATURES];
    private float[][][] weightedHistoryBuf = new float[1][HISTORY_STEPS][HISTORY_FEATURES];
    private void updateHistory(float[] newStep) {
        if (!historyInitialized) {
            for (int t = 0; t < HISTORY_STEPS; t++) {
                rawHistoryBuf[0][t] = newStep.clone();
            }
            historyInitialized = true;
        } else {
            //shift raw buffer
            for (int t = 0; t < HISTORY_STEPS - 1; t++) {
                rawHistoryBuf[0][t] = rawHistoryBuf[0][t + 1].clone();
            }
            rawHistoryBuf[0][HISTORY_STEPS - 1] = newStep.clone();
        }
        //apply recency to weighted buffer
        for (int t = 0; t < HISTORY_STEPS; t++) {
            float recency = (t + 1) / (float) HISTORY_STEPS;
            for (int f = 0; f < HISTORY_FEATURES; f++) {
                weightedHistoryBuf[0][t][f] = rawHistoryBuf[0][t][f] * recency;
            }
        }
    }


    double getHeading() {
        return imu.getRobotYawPitchRollAngles().getYaw(AngleUnit.RADIANS);
    }

    float[] convertCoordinates(float x, float y) {
        return new float[]{x, y};
    }

    private MappedByteBuffer loadModelFile(String modelPath) throws IOException {
        AssetFileDescriptor fileDescriptor =
                hardwareMap.appContext.getAssets().openFd(modelPath);
        FileInputStream inputStream =
                new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset   = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    public void drawField(Pose2d pose,
                          float relX, float relY,
                          float g1x,  float g1y,
                          float g2x,  float g2y) {
        TelemetryPacket packet = new TelemetryPacket();
        Canvas fieldOverlay = packet.fieldOverlay();
        DashboardUtil.drawRobot(fieldOverlay, pose);
        fieldOverlay.setStroke(COLOR_INACTIVE_TRAJECTORY);
        DashboardUtil.drawSampledPath(fieldOverlay, new Path(new PathSegment(
                new LineSegment(new Vector2d(relX, relY),
                        new Vector2d(g1x, g1y)))));
        DashboardUtil.drawSampledPath(fieldOverlay, new Path(new PathSegment(
                new LineSegment(new Vector2d(g1x, g1y),
                        new Vector2d(g2x, g2y)))));
        dashboard.sendTelemetryPacket(packet);
    }
}