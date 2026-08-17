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
@Autonomous(name = "AttentionGRU")
public class AttentionGRU extends LinearOpMode {

    IMU imu;
    TwoWheelTrackingLocalizer localizer;
    private ElapsedTime timer    = new ElapsedTime();
    private ElapsedTime runTimer = new ElapsedTime();
    private FtcDashboard dashboard;


    public static float[] goal1            = {20, 0, 0};
    public static float[] goal2            = {40, 0, 0};
    public static float[] startPos         = {0, 0, 0};


    public static float   GOAL_SWITCH_DIST1 = 5.0f;
    public static float   GOAL_SWITCH_DIST2 = 5.0f;
    public static float   VELOCITY_SCALE   = 1.0f;
    public static String  MODEL_FILE       = "Test4_crossATTN.tflite"; //Test1_HardSwitch, Test3_SigmoidBlend, Test4_crossATTN_NOGRU
    public static String  MODEL_NAME       = "AttentionGRU"; //Hard, Sigmoid, Attention
    public static int  ROUTE       = 4;
    public static int  SEED       = 3;

    private static final float POS_DIV = 72.0f;
    private static final float VEL_DIV = 30.0f;
    public static float LATENCY = 0;
    private static final float ANG_DIV = (float) Math.PI;

    private static final int HISTORY_STEPS    = 8;
    private static final int HISTORY_FEATURES = 3;
    private float[][][] rawHistoryBuf      = new float[1][HISTORY_STEPS][HISTORY_FEATURES];
    private float[][][] weightedHistoryBuf = new float[1][HISTORY_STEPS][HISTORY_FEATURES];
    private boolean historyInitialized     = false;

    private float prevVx    = 0;
    private float prevVy    = 0;
    private float prevOmega = 0;

    public static double kV      = 0.07;
    public static double kVTheta = 0.05;

    // ── M5 latency ────────────────────────────────────────────────
    private static final int   WARMUP_CALLS    = 200;
    private static final int   BENCHMARK_CALLS = 1000;
    private static final float DEADLINE_MS     = 10.0f;

    // ── M2/M3/M4 state ────────────────────────────────────────────
    private boolean goalSwitchFired     = false;
    private float   goalSwitchTime      = -1f;
    private float   preJumpVx           = 0;
    private float   preJumpVy           = 0;
    private float   preJumpOmega        = 0;
    private float   peakControlJump     = 0;     // M3
    private boolean goal1Reached        = false;
    private boolean goal2Reached        = false;

    // ── metrics log ───────────────────────────────────────────────
    // [time, relX, relY, heading, vx, vy, omega,
    //  outVx, outVy, outOmega, posError, crossTrackError,
    //  thetaError, velMag, goalSwitch, inferenceTime]
    private List<float[]> metricsLog = new ArrayList<>();

    @SuppressLint("DefaultLocale")
    @Override
    public void runOpMode() throws InterruptedException {

        // ── motors ────────────────────────────────────────────────
        DcMotorEx fl = hardwareMap.get(DcMotorEx.class, "fl");
        DcMotorEx bl = hardwareMap.get(DcMotorEx.class, "bl");
        DcMotorEx br = hardwareMap.get(DcMotorEx.class, "br");
        DcMotorEx fr = hardwareMap.get(DcMotorEx.class, "fr");

        fl.setMode(DcMotor.RunMode.RUN_WITHOUT_ENCODER);
        bl.setMode(DcMotor.RunMode.RUN_WITHOUT_ENCODER);
        br.setMode(DcMotor.RunMode.RUN_WITHOUT_ENCODER);
        fr.setMode(DcMotor.RunMode.RUN_WITHOUT_ENCODER);

        fl.setZeroPowerBehavior(DcMotor.ZeroPowerBehavior.BRAKE);
        bl.setZeroPowerBehavior(DcMotor.ZeroPowerBehavior.BRAKE);
        fr.setZeroPowerBehavior(DcMotor.ZeroPowerBehavior.BRAKE);
        br.setZeroPowerBehavior(DcMotor.ZeroPowerBehavior.BRAKE);

        fl.setDirection(DcMotorSimple.Direction.REVERSE);
        bl.setDirection(DcMotorSimple.Direction.REVERSE);
        br.setDirection(DcMotorSimple.Direction.FORWARD);
        fr.setDirection(DcMotorSimple.Direction.FORWARD);

        // ── TFLite ────────────────────────────────────────────────
        Interpreter interpreter;
        try {
            interpreter = new Interpreter(loadModelFile(MODEL_FILE));
        } catch (IOException e) {
            throw new RuntimeException("Failed to load model: " + e.getMessage());
        }

        telemetry.addData("Model inputs", interpreter.getInputTensorCount());
        for (int i = 0; i < interpreter.getInputTensorCount(); i++) {
            telemetry.addData("  input " + i,
                    interpreter.getInputTensor(i).name() +
                            " shape=" + Arrays.toString(interpreter.getInputTensor(i).shape()));
        }
        telemetry.update();

        // ── IMU + localizer ───────────────────────────────────────
        imu = hardwareMap.get(IMU.class, "imu");
        imu.initialize(new IMU.Parameters(new RevHubOrientationOnRobot(
                RevHubOrientationOnRobot.LogoFacingDirection.RIGHT,
                RevHubOrientationOnRobot.UsbFacingDirection.UP)));
        imu.resetYaw();
        localizer = new TwoWheelTrackingLocalizer(bl, fr,
                () -> AngleUnit.normalizeRadians(
                        imu.getRobotYawPitchRollAngles().getYaw(AngleUnit.RADIANS)),
                () -> imu.getRobotAngularVelocity(AngleUnit.RADIANS).zRotationRate);

        dashboard  = FtcDashboard.getInstance();
        Telemetry tel = new MultipleTelemetry(this.telemetry, dashboard.getTelemetry());

        // ── M5: run latency benchmark before main loop ────────────
        runM5LatencyBenchmark(interpreter, tel);

        float   goalSwitch = 0.0f;
        boolean done       = false;

        waitForStart();
        runTimer.reset();
        timer.reset();
        double loopTimer = timer.milliseconds();
        double startTimer = timer.milliseconds();
        float[] lastHist = {0.0f, 0.0f, 0.0f};

        while (opModeIsActive() && !done) {
            if(timer.milliseconds() < startTimer+LATENCY){
                continue;
            }
            else{
                startTimer = timer.milliseconds();
            }
            long loopStart = System.currentTimeMillis();

            localizer.update();
            float  elapsed = (float) runTimer.seconds();
            Pose2d pose    = localizer.getPoseEstimate();
            float  heading = (float) getHeading();
            float  sinTh   = (float) Math.sin(heading);
            float  cosTh   = (float) Math.cos(heading);

            float[] convPose  = convertCoordinates((float) pose.getX(), (float) pose.getY());
            float[] convStart = convertCoordinates(startPos[0], startPos[1]);
            float[] convG1    = convertCoordinates(goal1[0], goal1[1]);
            float[] convG2    = convertCoordinates(goal2[0], goal2[1]);

            float relX  = convPose[0]  - convStart[0];
            float relY  = convPose[1]  - convStart[1];
            float relG1X = convG1[0]   - convStart[0];
            float relG1Y = convG1[1]   - convStart[1];
            float relG2X = convG2[0]   - convStart[0];
            float relG2Y = convG2[1]   - convStart[1];

            float dist1 = dist(relX, relY, relG1X, relG1Y);
            float dist2 = dist(relX, relY, relG2X, relG2Y);

            // ── goalswitch + M2/M4 tracking ───────────────────────
            if (dist1 <= GOAL_SWITCH_DIST1 && goalSwitch == 0.0f) {
                goalSwitch    = 1.0f;
                goalSwitchFired = true;
                goalSwitchTime  = elapsed;
                goal1Reached    = true;   // M2/M4: goal1 successfully reached
                tel.addData("gs", "switched to goal2");
            }
            if (dist2 <= GOAL_SWITCH_DIST2 && goalSwitch == 1.0f) {
                goal2Reached = true;      // M4: goal2 successfully reached
                done = true;
            }


            // ── velocity ──────────────────────────────────────────
            Pose2d vel = localizer.getPoseVelocity();
            float vx    = 0, vy = 0, omega = 0;
            if (vel != null) {
                vx    = (float) vel.getX();
                vy    = (float) vel.getY();
                omega = (float) vel.getHeading();
            }
//            Pose2d velocitiesFC = fieldCentric(vx, vy, omega, heading);
//            vx = (float) velocitiesFC.getX();
//            vy = (float) velocitiesFC.getY();
//            omega = (float) velocitiesFC.getHeading();

            // ── build inputs ──────────────────────────────────────
            float[][] query = new float[1][8];
            query[0][0] = relX    / POS_DIV;
            query[0][1] = relY    / POS_DIV;
            query[0][2] = sinTh;
            query[0][3] = cosTh;
            query[0][4] = vx      / VEL_DIV;
            query[0][5] = vy      / VEL_DIV;
            query[0][6] = omega   / ANG_DIV;
            query[0][7] = goalSwitch;
            telemetry.addData("vx", vx);
            telemetry.addData("vy", vy);
            telemetry.addData("omega", omega);

            float[][][] waypoints = new float[1][2][4];
            waypoints[0][0][0] = relG1X / POS_DIV;
            waypoints[0][0][1] = relG1Y / POS_DIV;
            waypoints[0][0][2] = (float) Math.sin(goal1[2]);
            waypoints[0][0][3] = (float) Math.cos(goal1[2]);
            waypoints[0][1][0] = relG2X / POS_DIV;
            waypoints[0][1][1] = relG2Y / POS_DIV;
            waypoints[0][1][2] = (float) Math.sin(goal2[2]);
            waypoints[0][1][3] = (float) Math.cos(goal2[2]);

            float[][] gs = new float[1][1];
            gs[0][0] = goalSwitch;

            float[] newHistStep = new float[HISTORY_FEATURES];
            newHistStep[0] = (relX - lastHist[0])   / POS_DIV;
            newHistStep[1] = (relY - lastHist[1])    / POS_DIV;
            newHistStep[2] = (heading - lastHist[2]) / ANG_DIV;
//            newHistStep[3] = sinTh;
//            newHistStep[4] = cosTh;
//            newHistStep[5] = vx      / VEL_DIV;
//            newHistStep[6] = vy      / VEL_DIV;
//            newHistStep[7] = omega   / ANG_DIV;
//            newHistStep[5] = goalSwitch;

//            if(timer.milliseconds() >= loopTimer + 500){
            updateHistory(newHistStep);
//                loopTimer = timer.milliseconds();
//            }



            float[][] output = new float[1][3];
            Object[]  inputs = new Object[4];
            Map<Integer, Object> outputMap = new HashMap<>();
            outputMap.put(0, output);

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

            double startTime    = timer.milliseconds();
            interpreter.runForMultipleInputsOutputs(inputs, outputMap);
            double inferenceTime = timer.milliseconds() - startTime;


            float outVx    = output[0][0] * VEL_DIV;
            float outVy    = output[0][1] * VEL_DIV;
            float outOmega = output[0][2] * ANG_DIV;

            float alpha = 0.7f;   // higher = more smoothing
            outVx    = alpha * outVx    + (1 - alpha) * prevVx;
            outVy    = alpha * outVy    + (1 - alpha) * prevVy;
            outOmega = alpha * outOmega + (1 - alpha) * prevOmega;

            // ── M3: peak control jump at goalswitch ───────────────
            // record pre-switch command one step before switch fires
            if (!goalSwitchFired) {
                preJumpVx    = outVx;
                preJumpVy    = outVy;
                preJumpOmega = outOmega;
            }

            float jumpMag = 0.0f;
            if (goalSwitchFired && goalSwitchTime > 0 &&
                    elapsed - goalSwitchTime < 0.2f &&
                    elapsed - goalSwitchTime > 0f) {
                float jumpVx    = Math.abs(outVx    - preJumpVx);
                float jumpVy    = Math.abs(outVy    - preJumpVy);
                float jumpOmega = Math.abs(outOmega - preJumpOmega);
                jumpMag   = (float) Math.sqrt(
                        jumpVx*jumpVx + jumpVy*jumpVy + jumpOmega*jumpOmega);
                if (jumpMag > peakControlJump) peakControlJump = jumpMag;
            }

            // ── drive ─────────────────────────────────────────────
            List<Double> drivePowers = getDrivePower(
                    new Pose2d(outVx * kV, outVy * kV, outOmega * kVTheta));
            double flPower = drivePowers.get(0);
            double blPower = drivePowers.get(1);
            double brPower = drivePowers.get(2);
            double frPower = drivePowers.get(3);
            float  maxPow  = 1.0f;

            fl.setPower(flPower / maxPow);
            bl.setPower(blPower / maxPow);
            br.setPower(brPower / maxPow);
            fr.setPower(frPower / maxPow);

            prevVx    = outVx;
            prevVy    = outVy;
            prevOmega = outOmega;

            // ── metrics ───────────────────────────────────────────
            float activeGoalX    = (goalSwitch == 0) ? relG1X : relG2X;
            float activeGoalY    = (goalSwitch == 0) ? relG1Y : relG2Y;
            float posError       = dist(relX, relY, activeGoalX, activeGoalY);
            float crossTrackErr;
            if (goalSwitch == 0) {
                crossTrackErr = crossTrackError(relX, relY,
                        0, 0,
                        relG1X, relG1Y
                );
            } else {
                crossTrackErr = crossTrackError(relX, relY,
                        relG1X, relG1Y,
                        relG2X, relG2Y
                );
            }
            float targetHeading  = (goalSwitch == 0) ? goal1[2] : goal2[2];
            float thetaError     = Math.abs(normalizeAngle(heading - targetHeading));
            float velMag         = (float) Math.sqrt(outVx*outVx + outVy*outVy);

            metricsLog.add(new float[]{
                    elapsed,
                    relX, relY, heading,
                    vx, vy, omega,
                    outVx, outVy, outOmega,
                    posError, crossTrackErr,
                    thetaError, velMag,
                    goalSwitch, (float) inferenceTime, jumpMag
            });

            // ── telemetry ─────────────────────────────────────────
            tel.addData("inferenceTime (ms)", inferenceTime);
            tel.addData("goalSwitch",          goalSwitch);
            tel.addData("dist1 / dist2",
                    String.format("%.2f / %.2f", dist1, dist2));
            tel.addData("pose (rel)",
                    String.format("x=%.2f y=%.2f h=%.2f", relX, relY, heading));
            tel.addData("output (norm)",
                    String.format("vx=%.3f vy=%.3f om=%.3f",
                            output[0][0], output[0][1], output[0][2]));
            tel.addData("output (real)",
                    String.format("vx=%.3f vy=%.3f om=%.3f",
                            outVx, outVy, outOmega));
            tel.addData("M3 peak jump",
                    String.format("%.4f", peakControlJump));
            tel.addData("powers",
                    String.format("fl=%.2f bl=%.2f br=%.2f fr=%.2f",
                            flPower/maxPow, blPower/maxPow,
                            brPower/maxPow, frPower/maxPow));
            tel.update();

            drawField(pose, relX, relY, relG1X, relG1Y, relG2X, relG2Y);
            lastHist = new float[]{relX, relY, heading};

        }

        // ── stop motors ───────────────────────────────────────────
        fl.setPower(0);
        bl.setPower(0);
        br.setPower(0);
        fr.setPower(0);

        // ── save all diagnostics ──────────────────────────────────
        saveMetrics();
        saveM0Contract(tel);
        printSummary();
        tel.update();
        sleep(5000);
    }


    private void saveM0Contract(Telemetry tel) {
        if (metricsLog.isEmpty()) return;

        int   n           = metricsLog.size();
        float duration    = metricsLog.get(n-1)[0];

        // compute all summary stats
        float sumPosErr = 0, sumCross = 0, sumTheta = 0, sumInf = 0;
        float maxPosErr = 0, maxCross = 0;
        float sumInfSq  = 0;

        for (float[] row : metricsLog) {
            sumPosErr += row[10];
            sumCross  += row[11];
            sumTheta  += row[12];
            sumInf    += row[15];
            sumInfSq  += row[15] * row[15];
            if (row[10] > maxPosErr) maxPosErr = row[10];
            if (row[11] > maxCross)  maxCross  = row[11];
        }

        // M2: handoff failure k/N
        // handoff fails if goalswitch never fired or goal1 not reached
        int m2_k = goal1Reached ? 0 : 1;
        int m2_n = 1;   // single run

        // M3: peak control jump (computed during loop)
        float m3_jump = peakControlJump;

        // M4: completion k/N
        int m4_k = (goal1Reached && goal2Reached) ? 1 : 0;
        int m4_n = 1;

        // M5: inference stats from metricsLog
        float[] inferences = new float[n];
        for (int i = 0; i < n; i++) inferences[i] = metricsLog.get(i)[15];
        Arrays.sort(inferences);
        float inf_p50      = inferences[n / 2];
        float inf_p95      = inferences[(int)(n * 0.95f)];
        int   deadlineMiss = 0;
        for (float inf : inferences) if (inf > DEADLINE_MS) deadlineMiss++;


        //crosstrack
        float sumSquaredCTE = 0.0f;
        int cteCount = 0;
        float sumCTE = 0;

        for (float[] row : metricsLog) {
            float crossTrackErr = row[11];
            if (Float.isFinite(crossTrackErr)) {
                float squaredErr = crossTrackErr * crossTrackErr;
                sumCTE +=crossTrackErr;
                sumSquaredCTE += squaredErr;
                cteCount++;
            }
        }

        float crossTrackRMSE = (cteCount > 0)
                ? (float) Math.sqrt(sumSquaredCTE / cteCount)
                : Float.NaN;
        // Standard deviation of cross-track error
        float crossTrackMean = (cteCount > 0)
                ? sumCTE / cteCount
                : Float.NaN;

        float crossTrackVariance = (cteCount > 1)
                ? (sumSquaredCTE - cteCount * crossTrackMean * crossTrackMean)
                / (cteCount - 1)
                : Float.NaN;

        float crossTrackStdDev = (cteCount > 1)
                ? (float) Math.sqrt(Math.max(0.0, crossTrackVariance))
                : Float.NaN;

        String timestamp = new SimpleDateFormat(
                "yyyyMMdd_HHmmss", Locale.US).format(new Date());
        String filename  = MODEL_NAME + "_Summary_" + ROUTE + "_" + SEED + ".csv";

        File dir  = Environment.getExternalStoragePublicDirectory(
                Environment.DIRECTORY_DOWNLOADS);
        File file = new File(dir, filename);

        try {
            FileWriter w = new FileWriter(file);
            w.write("field,value\n");
            w.write(String.format(Locale.US, "model,%s\n",         MODEL_NAME));
            w.write(String.format(Locale.US, "model_file,%s\n",    MODEL_FILE));
            w.write(String.format(Locale.US, "timestamp,%s\n",     timestamp));
            w.write(String.format(Locale.US, "duration_s,%.3f\n",  duration));
            w.write(String.format(Locale.US, "steps,%d\n",         n));
            w.write(String.format(Locale.US, "goal1,%s\n",
                    Arrays.toString(goal1)));
            w.write(String.format(Locale.US, "goal2,%s\n",
                    Arrays.toString(goal2)));
            w.write(String.format(Locale.US,
                    "goal_switch_dist,%.2f\n",  GOAL_SWITCH_DIST1));
            w.write(String.format(Locale.US,
                    "goal_switch_time_s,%.3f\n", goalSwitchTime));
            // M2
            w.write(String.format(Locale.US,
                    "M2_handoff_fail_k,%d\n",   m2_k));
            w.write(String.format(Locale.US,
                    "M2_handoff_fail_N,%d\n",   m2_n));
            w.write(String.format(Locale.US,
                    "M2_handoff_fail_pct,%.1f\n", 100f * m2_k / m2_n));
            // M3
            w.write(String.format(Locale.US,
                    "M3_peak_control_jump,%.5f\n", m3_jump));
            // M4
            tel.addData("Peak Control Jump", m3_jump);
            w.write(String.format(Locale.US,
                    "M4_completion_k,%d\n",     m4_k));
            w.write(String.format(Locale.US,
                    "M4_completion_N,%d\n",     m4_n));
            w.write(String.format(Locale.US,
                    "M4_completion_pct,%.1f\n", 100f * m4_k / m4_n));
            // M5 — from control loop inference times
            w.write(String.format(Locale.US,
                    "M5_inf_p50_ms,%.3f\n",     inf_p50));
            w.write(String.format(Locale.US,
                    "M5_inf_p95_ms,%.3f\n",     inf_p95));
            w.write(String.format(Locale.US,
                    "M5_deadline_ms,%.1f\n",    DEADLINE_MS));
            w.write(String.format(Locale.US,
                    "M5_deadline_misses,%d\n",  deadlineMiss));
            w.write(String.format(Locale.US,
                    "M5_deadline_miss_pct,%.2f\n",
                    100f * deadlineMiss / n));
            // trajectory quality
            w.write(String.format(Locale.US,
                    "avg_pos_error_in,%.4f\n",  sumPosErr / n));
            w.write(String.format(Locale.US,
                    "max_pos_error_in,%.4f\n",  maxPosErr));
            w.write(String.format(Locale.US,
                    "avg_crosstrack_in,%.4f\n", sumCross / n));
            w.write(String.format(Locale.US,
                    "max_crosstrack_in,%.4f\n", maxCross));
            w.write(String.format(Locale.US,
                    "avg_theta_error_rad,%.4f\n", sumTheta / n));
            w.write(String.format(Locale.US,
                    "avg_inference_ms,%.3f\n",  sumInf / n));
            w.write(String.format(Locale.US,
                    "RMSE Cross Track,%.3f\n",  crossTrackRMSE));
            tel.addData("RMSE Cross Track,%.3f\n",  crossTrackRMSE);
            w.write(String.format(Locale.US,
                    "CT STD DEV,%.3f\n",  crossTrackStdDev));
            tel.addData("STD DEV\n",  crossTrackStdDev);

            w.flush();
            w.close();
            tel.addData("M0 saved", file.getAbsolutePath());
            tel.update();

        } catch (IOException e) {
            tel.addData("M0 save error", e.getMessage());
        }
    }



    private void runM5LatencyBenchmark(Interpreter interpreter,
                                       Telemetry tel) {
        tel.addData("M5", "Running latency benchmark...");
        tel.update();


        float[][] bQuery = new float[1][8];
        bQuery[0] = new float[]{
                0.1f, -0.1f, 0.7f, 0.7f, 0.3f, 0.1f, 0.05f, 0.0f};

        float[][][] bWp = new float[1][2][4];
        bWp[0][0] = new float[]{0.14f, -0.14f, 0.0f, 1.0f};
        bWp[0][1] = new float[]{0.28f, -0.28f, 0.0f, 1.0f};

        float[][][] bHist = new float[1][HISTORY_STEPS][HISTORY_FEATURES];
        for (int t = 0; t < HISTORY_STEPS; t++) {
            float r = (t + 1) / (float) HISTORY_STEPS;
            bHist[0][t] = new float[]{
                    0.1f*r, -0.1f*r, 0.05f*r};
//                    0.7f, 0.7f};
        }

        float[][] bGs    = new float[1][1];
        bGs[0][0]        = 0.0f;
        float[][] bOutput = new float[1][3];

        Object[] bInputs = new Object[4];
        Map<Integer, Object> bOutputMap = new HashMap<>();
        bOutputMap.put(0, bOutput);

        for (int i = 0; i < interpreter.getInputTensorCount(); i++) {
            int[] shape = interpreter.getInputTensor(i).shape();
            if (Arrays.equals(shape, new int[]{1, 8})) {
                bInputs[i] = bQuery;
            } else if (Arrays.equals(shape, new int[]{1, 2, 4})) {
                bInputs[i] = bWp;
            } else if (Arrays.equals(shape, new int[]{1,
                    HISTORY_STEPS, HISTORY_FEATURES})) {
                bInputs[i] = bHist;
            } else if (Arrays.equals(shape, new int[]{1, 1})) {
                bInputs[i] = bGs;
            }
        }


        tel.addData("M5", "Warmup (" + WARMUP_CALLS + ")...");
        tel.update();
        for (int i = 0; i < WARMUP_CALLS; i++) {
            runFullPipelineTimed(interpreter, bInputs, bOutputMap, bOutput);
        }


        tel.addData("M5", "Measuring (" + BENCHMARK_CALLS + ")...");
        tel.update();
        long[] lat_ns = new long[BENCHMARK_CALLS];
        for (int i = 0; i < BENCHMARK_CALLS; i++) {
            long t0  = System.nanoTime();
            runFullPipelineTimed(interpreter, bInputs, bOutputMap, bOutput);
            lat_ns[i] = System.nanoTime() - t0;
        }


        float[] lat_ms = new float[BENCHMARK_CALLS];
        for (int i = 0; i < BENCHMARK_CALLS; i++) {
            lat_ms[i] = lat_ns[i] / 1_000_000.0f;
        }
        Arrays.sort(lat_ms);

        float p50  = lat_ms[BENCHMARK_CALLS / 2];
        float p95  = lat_ms[(int)(BENCHMARK_CALLS * 0.95f)];
        float p99  = lat_ms[(int)(BENCHMARK_CALLS * 0.99f)];
        float lMin = lat_ms[0];
        float lMax = lat_ms[BENCHMARK_CALLS - 1];
        float mean = 0;
        for (float l : lat_ms) mean += l;
        mean /= BENCHMARK_CALLS;

        int violations = 0;
        for (float l : lat_ms) if (l > DEADLINE_MS) violations++;
        float vPct = 100f * violations / BENCHMARK_CALLS;


//        saveM5Results(lat_ms, p50, p95, p99,
//                lMin, lMax, mean, violations, vPct);

        tel.addData("── M5 LATENCY ──────────", "");
        tel.addData("model",          MODEL_NAME);
        tel.addData("warmup calls",   WARMUP_CALLS);
        tel.addData("measured calls", BENCHMARK_CALLS);
        tel.addData("deadline (ms)",
                String.format(Locale.US, "%.1f", DEADLINE_MS));
        tel.addData("p50 (ms)",
                String.format(Locale.US, "%.3f", p50));
        tel.addData("p95 (ms)",
                String.format(Locale.US, "%.3f", p95));
        tel.addData("p99 (ms)",
                String.format(Locale.US, "%.3f", p99));
        tel.addData("min (ms)",
                String.format(Locale.US, "%.3f", lMin));
        tel.addData("max (ms)",
                String.format(Locale.US, "%.3f", lMax));
        tel.addData("mean (ms)",
                String.format(Locale.US, "%.3f", mean));
        tel.addData("violations",
                String.format(Locale.US, "%d/%d (%.1f%%)",
                        violations, BENCHMARK_CALLS, vPct));
        tel.addData("RT feasible",
                vPct < 5f ? "YES" : "NO");
        tel.update();
        sleep(4000);
    }

    private void runFullPipelineTimed(Interpreter interp,
                                      Object[] inputs,
                                      Map<Integer, Object> outputMap,
                                      float[][] output) {
        // 1. TFLite inference
        interp.runForMultipleInputsOutputs(inputs, outputMap);

        // 2. decode + denormalize
        float outVx    = output[0][0] * VEL_DIV;
        float outVy    = output[0][1] * VEL_DIV;
        float outOmega = output[0][2] * ANG_DIV;

        // 3. clip
        outVx    = Math.max(-VEL_DIV, Math.min(VEL_DIV, outVx));
        outVy    = Math.max(-VEL_DIV, Math.min(VEL_DIV, outVy));
        outOmega = Math.max(-ANG_DIV, Math.min(ANG_DIV, outOmega));

        // 4. command construction
        getDrivePower(new Pose2d(
                outVx * kV, outVy * kV, outOmega * kVTheta));
    }

    private void saveM5Results(float[] lat_ms,
                               float p50, float p95, float p99,
                               float lMin, float lMax, float mean,
                               int violations, float vPct) {
        String timestamp = new SimpleDateFormat(
                "yyyyMMdd_HHmmss", Locale.US).format(new Date());
        String filename  = "M5_latency_" + MODEL_NAME +".csv";
        File dir  = Environment.getExternalStoragePublicDirectory(
                Environment.DIRECTORY_DOWNLOADS);
        File file = new File(dir, filename);

        try {
            FileWriter w = new FileWriter(file);

            // summary block
            w.write("metric,value\n");
            w.write(String.format(Locale.US, "model,%s\n",
                    MODEL_NAME));
            w.write(String.format(Locale.US, "warmup_calls,%d\n",
                    WARMUP_CALLS));
            w.write(String.format(Locale.US, "measured_calls,%d\n",
                    BENCHMARK_CALLS));
            w.write(String.format(Locale.US, "deadline_ms,%.1f\n",
                    DEADLINE_MS));
            w.write(String.format(Locale.US, "p50_ms,%.4f\n",   p50));
            w.write(String.format(Locale.US, "p95_ms,%.4f\n",   p95));
            w.write(String.format(Locale.US, "p99_ms,%.4f\n",   p99));
            w.write(String.format(Locale.US, "min_ms,%.4f\n",   lMin));
            w.write(String.format(Locale.US, "max_ms,%.4f\n",   lMax));
            w.write(String.format(Locale.US, "mean_ms,%.4f\n",  mean));
            w.write(String.format(Locale.US, "violations,%d\n",
                    violations));
            w.write(String.format(Locale.US, "violation_pct,%.2f\n",
                    vPct));

            // raw latency series
            w.write("\ncall_index,latency_ms\n");
            for (int i = 0; i < lat_ms.length; i++) {
                w.write(String.format(Locale.US,
                        "%d,%.4f\n", i, lat_ms[i]));
            }

            w.flush();
            w.close();
            telemetry.addData("M5 saved", file.getAbsolutePath());

        } catch (IOException e) {
            telemetry.addData("M5 error", e.getMessage());
        }
    }


    private float dist(float x1, float y1, float x2, float y2) {
        return (float) Math.sqrt(
                Math.pow(x2-x1, 2) + Math.pow(y2-y1, 2));
    }

    private float crossTrackError(float px, float py,
                                  float ax, float ay,
                                  float bx, float by) {
        float dx  = bx - ax;
        float dy  = by - ay;
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
        String filename  = MODEL_NAME + "_Full_" + ROUTE + "_" + SEED + ".csv";
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
                            "goalswitch,inference_ms,"+
                            "jumpMag\n"
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
            telemetry.addData("metrics saved", file.getAbsolutePath());

        } catch (IOException e) {
            telemetry.addData("save error", e.getMessage());
        }
    }

    private void printSummary() {
        if (metricsLog.isEmpty()) return;
        int   n           = metricsLog.size();
        float totalPosErr = 0, totalCross = 0,
                totalTheta  = 0, totalInf   = 0;
        float maxPosErr   = 0, maxCross   = 0;
        float duration    = metricsLog.get(n-1)[0];

        for (float[] row : metricsLog) {
            totalPosErr += row[10]; totalCross += row[11];
            totalTheta  += row[12]; totalInf   += row[15];
            if (row[10] > maxPosErr) maxPosErr = row[10];
            if (row[11] > maxCross)  maxCross  = row[11];
        }

        telemetry.addData("── SUMMARY ──────────────", "");
        telemetry.addData("model",         MODEL_NAME);
        telemetry.addData("duration (s)",
                String.format(Locale.US, "%.2f", duration));
        telemetry.addData("steps",         n);
        telemetry.addData("M2 handoff",
                goal1Reached ? "SUCCESS" : "FAIL");
        telemetry.addData("M3 peak jump",
                String.format(Locale.US, "%.4f", peakControlJump));
        telemetry.addData("M4 completion",
                (goal1Reached && goal2Reached) ? "SUCCESS" : "FAIL");
        telemetry.addData("avg pos error",
                String.format(Locale.US, "%.3f in", totalPosErr / n));
        telemetry.addData("max pos error",
                String.format(Locale.US, "%.3f in", maxPosErr));
        telemetry.addData("avg cross-track",
                String.format(Locale.US, "%.3f in", totalCross / n));
        telemetry.addData("max cross-track",
                String.format(Locale.US, "%.3f in", maxCross));
        telemetry.addData("avg theta error",
                String.format(Locale.US, "%.3f rad", totalTheta / n));
        telemetry.addData("avg inference",
                String.format(Locale.US, "%.2f ms", totalInf / n));
        telemetry.update();
    }
    float[] prevStep = new float[HISTORY_FEATURES];
    private void updateHistory(float[] newStep) {
        if (!historyInitialized) {
            for (int t = 0; t < HISTORY_STEPS; t++) {
                rawHistoryBuf[0][t] = new float[HISTORY_FEATURES];
            }
            historyInitialized = true;
        } else {
            for (int t = 0; t < HISTORY_STEPS - 1; t++) {
                rawHistoryBuf[0][t] = rawHistoryBuf[0][t + 1].clone();
            }
            rawHistoryBuf[0][HISTORY_STEPS - 1] = prevStep.clone();
        }
        for (int t = 0; t < HISTORY_STEPS; t++) {
//            float recency = (t + 1) / (float) HISTORY_STEPS;
            for (int f = 0; f < HISTORY_FEATURES; f++) {
                weightedHistoryBuf[0][t][f] =
                        rawHistoryBuf[0][t][f];//* recency;
            }
        }
        prevStep = newStep;
    }

    public Pose2d fieldCentric(double x, double y,
                               double h, double currHeading) {
        Vector2d vec = new Vector2d(x, y).rotated(-currHeading);
        return new Pose2d(vec.getX(), vec.getY(), h);
    }

    double getHeading() {
        return imu.getRobotYawPitchRollAngles().getYaw(AngleUnit.RADIANS);
    }

    float[] convertCoordinates(float x, float y) {
        return new float[]{x, y};
    }

    private MappedByteBuffer loadModelFile(String modelPath)
            throws IOException {
        AssetFileDescriptor fd =
                hardwareMap.appContext.getAssets().openFd(modelPath);
        FileInputStream fis = new FileInputStream(fd.getFileDescriptor());
        FileChannel     fc  = fis.getChannel();
        return fc.map(FileChannel.MapMode.READ_ONLY,
                fd.getStartOffset(), fd.getDeclaredLength());
    }

    public void drawField(Pose2d pose,
                          float relX,  float relY,
                          float g1x,   float g1y,
                          float g2x,   float g2y) {
        TelemetryPacket packet = new TelemetryPacket();
        Canvas fieldOverlay    = packet.fieldOverlay();
        DashboardUtil.drawRobot(fieldOverlay, pose);
        fieldOverlay.setStroke(COLOR_INACTIVE_TRAJECTORY);
        DashboardUtil.drawSampledPath(fieldOverlay, new Path(
                new PathSegment(new LineSegment(
                        new Vector2d(relX, relY),
                        new Vector2d(g1x, g1y)))));
        DashboardUtil.drawSampledPath(fieldOverlay, new Path(
                new PathSegment(new LineSegment(
                        new Vector2d(g1x, g1y),
                        new Vector2d(g2x, g2y)))));
        dashboard.sendTelemetryPacket(packet);
    }

    private List<Double> getDrivePower(Pose2d pose2d) {
        return MecanumKinematics.robotToWheelVelocities(
                pose2d, 12, 12, 1);
    }
}