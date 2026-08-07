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
import com.acmerobotics.roadrunner.kinematics.MecanumKinematics;
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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import android.os.Environment;

import java.io.File;
import java.io.FileWriter;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

import static org.firstinspires.ftc.teamcode.trajectorysequence.TrajectorySequenceRunner.COLOR_INACTIVE_TRAJECTORY;

@Config
@Autonomous(name = "TwoInputTest")
public class TwoInputTest0 extends LinearOpMode {
    IMU imu;
    TwoWheelTrackingLocalizer localizer;
    private ElapsedTime timer = new ElapsedTime();
    private ElapsedTime runTimer = new ElapsedTime();
    private FtcDashboard dashboard;

    public static float[] goal1    = {10, 10, 1};  // x, y, heading
    public static float[] goal2    = {-20, -40, -1};
    public static float[] startPos = {0, 0, 0};
    public static float GOAL_SWITCH_DIST = 5.0f;    // inches
    public static float VELOCITY_SCALE = 1.0f;    // scale factor on output velocities
    public static String MODEL_FILE = "Test0_Baseline.tflite";
    private static final float POS_DIV = 72.0f;
    private static final float VEL_DIV = 30.0f;
    private static final float ANG_DIV = (float) Math.PI;
    private boolean historyInitialized = false;
    private float prevVx    = 0;
    private float prevVy    = 0;
    private float prevOmega = 0;
    public static double  kV = 0.25;
    public static double kVTheta = 0.15;

    // each row: [timestamp, relX, relY, heading, vx, vy, omega,
    //            outVx, outVy, outOmega, posError, thetaError,
    //            goalSwitch, inferenceTime]
    private List<float[]> metricsLog = new ArrayList<>();

    @SuppressLint("DefaultLocale")
    @Override
    public void runOpMode() throws InterruptedException {
        //config so i dont forget:
        //control hub
        //0, fl
        //1, br
        //2, bl
        //3, fr
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
        br.setDirection(DcMotorSimple.Direction.FORWARD);
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
        localizer = new TwoWheelTrackingLocalizer(bl, fr,
                () -> AngleUnit.normalizeRadians(
                        imu.getRobotYawPitchRollAngles().getYaw(AngleUnit.RADIANS)),
                () -> imu.getRobotAngularVelocity(AngleUnit.RADIANS).zRotationRate);

        dashboard = FtcDashboard.getInstance();
        Telemetry telemetry = new MultipleTelemetry(this.telemetry, dashboard.getTelemetry());

        float goalSwitch = 0.0f;
        boolean done = false;

        waitForStart();
        runTimer.reset();
        timer.reset();

        while (opModeIsActive() && !done) {
            long loopStart = System.currentTimeMillis();
            localizer.update();
            float elapsed = (float) runTimer.seconds();
            Pose2d pose    = localizer.getPoseEstimate();

            float heading  = (float) getHeading();
            float sinTh    = (float) Math.sin(heading);
            float cosTh    = (float) Math.cos(heading);

            float[] convPose  = convertCoordinates((float) pose.getX(), (float) pose.getY());
            float[] convStart = convertCoordinates(startPos[0], startPos[1]);
            float[] convG1 = convertCoordinates(goal1[0], goal1[1]);
            float[] convG2 = convertCoordinates(goal2[0], goal2[1]);

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

            //output (1, 3) [vx_norm, vy_norm, omega_norm]
            float[][] output = new float[1][3];

            //inputs must match TFLite index order
            //verify order from telemetry printout at init
            Object[] inputs = new Object[2];
            Map<Integer, Object> outputs = new HashMap<>();
            outputs.put(0, output);

            //set by shape to handle any reordering
            for (int i = 0; i < interpreter.getInputTensorCount(); i++) {
                int[] shape = interpreter.getInputTensor(i).shape();
                if (Arrays.equals(shape, new int[]{1, 8})) {
                    inputs[i] = query;
                } else if (Arrays.equals(shape, new int[]{1, 2, 4})) {
                    inputs[i] = waypoints;
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



            //mecanum velocity
//            float flPower = VELOCITY_SCALE * (outVx - outVy - outOmega);
//            float blPower = VELOCITY_SCALE * (outVx
//            + outVy - outOmega);
//            float brPower = VELOCITY_SCALE * (outVx - outVy + outOmega);
//            float frPower = VELOCITY_SCALE * (outVx + outVy + outOmega);

//            float alpha = 0.7f;   // higher = more smoothing
//            outVx    = alpha * outVx    + (1 - alpha) * prevVx;
//            outVy    = alpha * outVy    + (1 - alpha) * prevVy;
//            outOmega = alpha * outOmega + (1 - alpha) * prevOmega;
            List<Double> drivePowers = getDrivePower(new Pose2d(outVx * kV, outVy * kV, outOmega * kVTheta));
//            List<Double> drivePowers = getDrivePower(new Pose2d(1, 1, 0.1)); //testing
            double flPower = drivePowers.get(0);
            double blPower = drivePowers.get(1);
            double brPower = drivePowers.get(2);
            double frPower = drivePowers.get(3);
            //normalize if any power exceeds 1.0
//            float maxPower = Math.max(1.0f,
//                    Math.max(Math.abs(flPower),
//                            Math.max(Math.abs(blPower),
//                                    Math.max(Math.abs(brPower), Math.abs(frPower)))));
            float maxPower = 1;
            fl.setPower(flPower / maxPower);
            bl.setPower(blPower / maxPower);
            br.setPower(brPower / maxPower);
            fr.setPower(frPower / maxPower);

            prevVx = outVx;
            prevVy = outVy;
            prevOmega = outOmega;

            float activeGoalX = (goalSwitch == 0) ? relG1X : relG2X;
            float activeGoalY = (goalSwitch == 0) ? relG1Y : relG2Y;
            float posError = dist(relX, relY, activeGoalX, activeGoalY);

            float crossTrackError = crossTrackError(relX, relY,
                    relG1X, relG1Y,
                    relG2X, relG2Y);

            float targetHeading = (goalSwitch == 0) ? goal1[2] : goal2[2];
            float thetaError    = Math.abs(normalizeAngle(heading - targetHeading));

            float velMag = (float) Math.sqrt(outVx*outVx + outVy*outVy);

            // [time, relX, relY, heading, vx, vy, omega,
            //  outVx, outVy, outOmega, posError, crossTrackError,
            //  thetaError, velMag, goalSwitch, inferenceTime]
            metricsLog.add(new float[]{
                    elapsed,
                    relX, relY, heading,
                    vx, vy, omega,
                    outVx, outVy, outOmega,
                    posError, crossTrackError,
                    thetaError, velMag,
                    goalSwitch, (float) inferenceTime
            });
            telemetry.addData("max power", maxPower);

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
            long elapsed1 = System.currentTimeMillis() - loopStart;
//            if (elapsed1 < 200) sleep(200 - elapsed1);
        }
        fl.setPower(0);
        fr.setPower(0);
        br.setPower(0);
        bl.setPower(0);
        saveMetrics();
        printSummary();
        telemetry.update();
        sleep(5000);
    }
    private float dist(float x1, float y1, float x2, float y2) {
        return (float) Math.sqrt(Math.pow(x2-x1, 2) + Math.pow(y2-y1, 2));
    }

    private float crossTrackError(float px, float py,
                                  float ax, float ay,
                                  float bx, float by) {
        //perpendicular distance
        float dx = bx - ax;
        float dy = by - ay;
        float len = (float) Math.sqrt(dx*dx + dy*dy);
        if (len < 1e-6f) return dist(px, py, ax, ay);
        return Math.abs(dy*px - dx*py + bx*ay - by*ax) / len;
    }

    private float normalizeAngle(float angle) {
        while (angle >  Math.PI) angle -= 2 * Math.PI;
        while (angle < -Math.PI) angle += 2 * Math.PI;
        return angle;
    }
    private void saveMetrics() {
        String timestamp = new SimpleDateFormat(
                "yyyyMMdd_HHmmss", Locale.US).format(new Date());
        String filename  = MODEL_FILE + "_" + timestamp + ".csv";

        File dir  = Environment.getExternalStoragePublicDirectory(
                Environment.DIRECTORY_DOWNLOADS);
        File file = new File(dir, filename);

        try {
            FileWriter writer = new FileWriter(file);
            writer.write(
                    "time_s,relX,relY,heading," +
                            "vx_in,vy_in,omega_in," +
                            "vx_out,vy_out,omega_out," +
                            "pos_error,cross_track_error," +
                            "theta_error,vel_mag," +
                            "goalswitch,inference_ms\n"
            );

            for (float[] row : metricsLog) {
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < row.length; i++) {
                    sb.append(String.format(Locale.US, "%.5f", row[i]));
                    if (i < row.length - 1) sb.append(",");
                }
                sb.append("\n");
                writer.write(sb.toString());
            }

            writer.flush();
            writer.close();
            telemetry.addData("saved to", file.getAbsolutePath());

        } catch (IOException e) {
            telemetry.addData("save error", e.getMessage());
        }
    }

    private void printSummary() {
        if (metricsLog.isEmpty()) return;

        int n = metricsLog.size();

        // compute summary stats
        float totalPosErr   = 0;
        float totalCrossErr = 0;
        float totalThetaErr = 0;
        float totalInfTime  = 0;
        float maxPosErr     = 0;
        float maxCrossErr   = 0;
        float totalDuration = metricsLog.get(n-1)[0];  // last timestamp

        for (float[] row : metricsLog) {
            float posErr   = row[10];
            float crossErr = row[11];
            float thetaErr = row[12];
            float infTime  = row[15];

            totalPosErr   += posErr;
            totalCrossErr += crossErr;
            totalThetaErr += thetaErr;
            totalInfTime  += infTime;

            if (posErr   > maxPosErr)   maxPosErr   = posErr;
            if (crossErr > maxCrossErr) maxCrossErr = crossErr;
        }

        telemetry.addData("── SUMMARY ──────────────", "");
        telemetry.addData("model", MODEL_FILE);
        telemetry.addData("duration (s)", String.format(Locale.US, "%.2f", totalDuration));
        telemetry.addData("steps logged", n);
        telemetry.addData("avg pos error",
                String.format(Locale.US, "%.3f in", totalPosErr / n));
        telemetry.addData("max pos error",
                String.format(Locale.US, "%.3f in", maxPosErr));
        telemetry.addData("avg cross-track",
                String.format(Locale.US, "%.3f in", totalCrossErr / n));
        telemetry.addData("max cross-track",
                String.format(Locale.US, "%.3f in", maxCrossErr));
        telemetry.addData("avg theta error",
                String.format(Locale.US, "%.3f rad", totalThetaErr / n));
        telemetry.addData("avg inference",
                String.format(Locale.US, "%.2f ms", totalInfTime / n));
        telemetry.update();
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
    private List<Double> getDrivePower(Pose2d pose2d) {
        return MecanumKinematics.robotToWheelVelocities(pose2d, 12, 12, 1);
    }
}