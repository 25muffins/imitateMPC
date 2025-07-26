package org.firstinspires.ftc.teamcode;

import static org.firstinspires.ftc.teamcode.trajectorysequence.TrajectorySequenceRunner.COLOR_INACTIVE_TRAJECTORY;

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
import com.qualcomm.robotcore.hardware.VoltageSensor;
import org.firstinspires.ftc.robotcore.external.Telemetry;
import org.firstinspires.ftc.robotcore.external.navigation.Pose3D;
import org.firstinspires.ftc.teamcode.drive.DriveConstants;
import org.firstinspires.ftc.teamcode.util.DashboardUtil;
import org.tensorflow.lite.Interpreter;


import org.firstinspires.ftc.robotcore.external.navigation.AngleUnit;
import org.firstinspires.ftc.teamcode.drive.TwoWheelTrackingLocalizer;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

@Config
@Autonomous(name = "MPCPathGen")
public class MPCPathGen extends LinearOpMode {
    IMU imu;
    TwoWheelTrackingLocalizer localizer;

    private ElapsedTime timer = new ElapsedTime();
    private FtcDashboard dashboard;
//for testing directions
//    public static double flSpeed = 0;
//    public static double blSpeed = 0;
//    public static double brSpeed = 0;
//    public static double frSpeed = 0;

    public static double speed = 3;
    public static double speedX = 1;
    public static double speedY = 1;
    public static double speedTheta = 1;
    public static double speed2 = 0;
    public static double distCutoff = 6;

    //for tuning feedforward
//    public static float veloX = 0;
//    public static float veloY  = 0;
//    public static float veloTheta = 0;


    public static float[] goal1 = {10, -10, 2};
    public static float[] goal2 = {20,-20, 2};
    public static float[] startPos = {0,0,0};
    public  static double[] fc = new double[]{0,  1, 0};
    public static double lateralMult = 1.7;
    public static double  kV = 0.015;
    public static double kVTheta = 0.2;
    public static double kA = 0;


    @Override
    public void runOpMode() throws InterruptedException {

        DcMotorEx fl = hardwareMap.get(DcMotorEx.class, "fl");
        DcMotorEx bl = hardwareMap.get(DcMotorEx.class, "bl");
        DcMotorEx br = hardwareMap.get(DcMotorEx.class, "br");
        DcMotorEx fr = hardwareMap.get(DcMotorEx.class, "fr");

        VoltageSensor voltageSensor = hardwareMap.voltageSensor.iterator().next();

        fl.setMode(DcMotor.RunMode.RUN_WITHOUT_ENCODER);
        bl.setMode(DcMotor.RunMode.RUN_WITHOUT_ENCODER);
        br.setMode(DcMotor.RunMode.RUN_WITHOUT_ENCODER);
        fr.setMode(DcMotor.RunMode.RUN_WITHOUT_ENCODER);

        fl.setDirection(DcMotorSimple.Direction.REVERSE);
        bl.setDirection(DcMotorSimple.Direction.REVERSE);
        br.setDirection(DcMotorSimple.Direction.REVERSE);
        fr.setDirection(DcMotorSimple.Direction.FORWARD);

        Interpreter interpreter;
        try  {
            interpreter = new Interpreter(loadModelFile());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        float[] inputs = {0,0,0, 0, 0, 0, 0, 0, 0, 0};
        float[][] outputs = {{0,0,0}};
        float goalSwitch = 0;
        float[] lastVel = {0, 0, 0}; // vx, vy, omega

        Pose2d pose;
        double heading;
        imu = hardwareMap.get(IMU.class, "imu");
        imu.initialize(new IMU.Parameters(new RevHubOrientationOnRobot(
                RevHubOrientationOnRobot.LogoFacingDirection.RIGHT,
                RevHubOrientationOnRobot.UsbFacingDirection.UP
        )));
        imu.resetYaw();
        localizer = new TwoWheelTrackingLocalizer(fl, bl,
                () -> AngleUnit.normalizeRadians(imu.getRobotYawPitchRollAngles().getYaw(AngleUnit.RADIANS)),
                () -> imu.getRobotAngularVelocity(AngleUnit.RADIANS).zRotationRate); //parallel, perp, orientation, velocity

        dashboard = FtcDashboard.getInstance();
        Telemetry telemetry = new MultipleTelemetry(this.telemetry, dashboard.getTelemetry());
        telemetry.addData("updated",  "updated");
        telemetry.update();
        drawField(new Pose2d(0,0,0), inputs);


        waitForStart();
        while(opModeIsActive()){
            double startTime = timer.milliseconds();
            pose = localizer.getPoseEstimate();
            inputs = new float[]{(float) pose.getX(), (float) pose.getY(), (float) getHeading(),
                    goal1[0],  goal1[1], goal1[2],  goal2[0], goal2[1], goal2[2], 0}; //100% just for field drawing
            drawField(pose, inputs); //field coordinates (weird one)


            //mpc coordinates (normal coordinates)
            float[] ConvertedStartXY = convertCoordinates(startPos[0], startPos[1]);
            float[] ConvertedPose = convertCoordinates((float) pose.getX(), (float) pose.getY());
            float[] ConvertedGoal1 = convertCoordinates(goal1[0], goal1[1]);
            float[] ConvertedGoal2 = convertCoordinates(goal2[0], goal2[1]);

            //relative coordinates
            float[] RelativePose = {ConvertedPose[0]  - ConvertedStartXY[0], ConvertedPose[1]  - ConvertedStartXY[1]};
            float[] RelativeGoal1 = {ConvertedGoal1[0] - ConvertedStartXY[0], ConvertedGoal1[1] - ConvertedStartXY[1]};
            float[] RelativeGoal2 = {ConvertedGoal2[0] - ConvertedStartXY[0], ConvertedGoal2[1] -  ConvertedStartXY[1]};
            float Dist1 = (float) Math.sqrt(Math.pow(RelativeGoal1[0] - RelativePose[0], 2) +  Math.pow(RelativeGoal1[1] - RelativePose[1],2));
            float Dist2 = (float) Math.sqrt(Math.pow(RelativeGoal2[0] - RelativePose[0], 2) +  Math.pow(RelativeGoal2[1] - RelativePose[1],2));

            if(Dist1 <= distCutoff && goalSwitch != 1){
                goalSwitch =  1;
            }
            if(Dist2 <=2 && goalSwitch == 1){
                speed = speed2;
            }

            inputs[0] = RelativePose[0];
            inputs[1] = RelativePose[1];

//            float HeadingChange = (float) (pose.getHeading() - Math.PI/4); //idk why i think it has something to do with the switched x and y
//            if(HeadingChange > Math.PI) { inputs[2] = (float) (HeadingChange - 2*Math.PI);}
//            else{ }
            inputs[2] = (float) getHeading();
            inputs[3] = RelativeGoal1[0];
            inputs[4] = RelativeGoal1[1];
            inputs[5] = goal1[2];
            inputs[6] = RelativeGoal2[0];
            inputs[7] = RelativeGoal2[1];
            inputs[8] = goal2[2];
            inputs[9] = goalSwitch;
            float[] normedInputs = normalizeInputs(inputs);


            timer.reset();

            interpreter.run(normedInputs, outputs);

            // [fr, fl, br, bl]


//            fr.setVelocity(velo*outputs[0][0]);
//            fl.setVelocity(velo*outputs[0][1]);
//            br.setVelocity(velo*outputs[0][2]);
//            bl.setVelocity(velo*outputs[0][3]);
            //for testing directions
//            fl.setPower(flSpeed);
//            bl.setPower(blSpeed);
//            br.setPower(brSpeed);
//            fr.setPower(frSpeed);

            telemetry.addData("outputs", Arrays.toString(outputs[0]));
            telemetry.addData("currentPosRelative", Arrays.toString(new float[]{inputs[0], inputs[1], inputs[2]}));
            telemetry.addData("goal1Relative", Arrays.toString(new float[]{RelativeGoal1[0], RelativeGoal1[1], goal1[2]}));
            telemetry.addData("goal2Relative", Arrays.toString(new float[]{RelativeGoal2[0], RelativeGoal2[1], goal2[2]}));
            telemetry.addData("inputs", Arrays.toString(normedInputs));
            telemetry.addData("imu", getHeading());
            telemetry.addData("goalswtich", goalSwitch);
            telemetry.addData("voltage", voltageSensor.getVoltage());
//            Pose2d fcTest = fieldCentric(fc[0], fc[1], fc[2], voltageSensor.getVoltage(), getHeading());
//            List<Double> dpTest = getDrivePower(fcTest);
//            telemetry.addData("fcTest", Arrays.toString(new double[]{fcTest.getX(), fcTest.getY(), fcTest.getHeading()}));
//            telemetry.addData("dpTest", Arrays.toString(new double[]{dpTest.get(0), dpTest.get(1), dpTest.get(2), dpTest.get(3)}));

            float[] convertBackOutputs = convertBack(outputs[0]);
            float vx = (float) (convertBackOutputs[0] * speed * speedX);
            float vy = (float) (convertBackOutputs[1] * speed * speedY);
            float omega = (float) (convertBackOutputs[2] * speed * speedTheta);

            //currently broken, not using
            double dt = 0.01;
            float ax = (convertBackOutputs[0] - lastVel[0]) / (float) dt;
            float ay = (convertBackOutputs[1] - lastVel[1]) / (float) dt;
            float aOmega = (convertBackOutputs[2] - lastVel[2]) / (float) dt;


            float ffX = (float)(kV * vx + kA * ax);
            float ffY = (float)(kV * vy + kA * ay);
            float ffOmega = (float)(kVTheta * omega + kA * aOmega);

            Pose2d fcActual = fieldCentric(ffX, ffY, ffOmega, voltageSensor.getVoltage(), getHeading());
            List<Double> dp = getDrivePower(fcActual);

            lastVel[0] = vx;
            lastVel[1] = vy;
            lastVel[2] = omega;

            telemetry.addData("vx", vx);
            telemetry.addData("vy", vy);
            telemetry.addData("omega", omega);

            telemetry.addData("ax", ax);
            telemetry.addData("ay", ay);
            telemetry.addData("aomega", aOmega);
            telemetry.addData("fcActual", Arrays.toString(new double[]{fcActual.getX(), fcActual.getY(), fcActual.getHeading()}));
            telemetry.addData("dp", Arrays.toString(new double[]{dp.get(0), dp.get(1), dp.get(2), dp.get(3)}));
//            bl.setPower(speed*dpTest.get(0));
//            fl.setPower(speed*dpTest.get(1));
//            fr.setPower(speed*dpTest.get(2));
//            br.setPower(speed*dpTest.get(3));
            fl.setPower(dp.get(0));
            bl.setPower(dp.get(1));
            br.setPower(dp.get(2));
            fr.setPower(dp.get(3));
            localizer.update();
            Pose2d poseVelo = Objects.requireNonNull(localizer.getPoseVelocity(), "poseVelocity() must not be null. Ensure that the getWheelVelocities() method has been overridden in your localizer.");
            telemetry.addData("veloX", poseVelo.getX());
            telemetry.addData("veloY", -poseVelo.getY());
            telemetry.addData("veloHeading", poseVelo.getHeading());

            double totalTime = timer.milliseconds() - startTime;
            telemetry.addData("totalTime", totalTime);
            telemetry.update();

        }
    }
    double getHeading() {
        return imu.getRobotYawPitchRollAngles().getYaw(AngleUnit.RADIANS);
    }
    float[] convertCoordinates(float x, float y){
        return new float[] {-y, x};
    }
    float[] convertBack(float[] outputs){
        return new float[] {outputs[1]*30, -outputs[0]*30, (float) (outputs[2]*3.1415)};
    }
    private MappedByteBuffer loadModelFile() throws IOException {
        String modelPath = "MPCPathGen.tflite";
        AssetFileDescriptor fileDescriptor =
                hardwareMap.appContext.getAssets().openFd(modelPath);
        FileInputStream inputStream =
                new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
    public void drawField(Pose2d pose, float[] inputs){
        TelemetryPacket packet = new TelemetryPacket();
        Canvas fieldOverlay = packet.fieldOverlay();
        DashboardUtil.drawRobot(fieldOverlay, pose);
        fieldOverlay.setStroke(COLOR_INACTIVE_TRAJECTORY);
        DashboardUtil.drawSampledPath(fieldOverlay, new Path(new PathSegment(
                new LineSegment(new Vector2d(inputs[0], inputs[1]),
                        new Vector2d(goal1[0], goal1[1])))));
        DashboardUtil.drawSampledPath(fieldOverlay, new Path(new PathSegment(
                new LineSegment(new Vector2d(goal1[0], goal1[1]),
                        new Vector2d(goal2[0], goal2[1])))));
        dashboard.sendTelemetryPacket(packet);
    }
    public float[] normalizeInputs(float[] inputs){
        float[] normInputs = {1,2,3,4,5,6,7,8,9,10};
        normInputs[0] = inputs[0]/72;
        normInputs[1] = inputs[1]/72;
        normInputs[2] = (float) (inputs[2]/3.1415);
        normInputs[3] = inputs[3]/72;
        normInputs[4] = inputs[4]/72;
        normInputs[5] = (float) (inputs[5]/3.1415);
        normInputs[6] = inputs[6]/72;
        normInputs[7] = inputs[7]/72;
        normInputs[8] = (float) (inputs[8]/3.1415);
        normInputs[9] = inputs[9];
        return normInputs;
    }
    public Pose2d fieldCentric(double x, double y, double h, double voltage, double currHeading) {
        float voltageComp = (float) (12.0 / voltage);
        double[] p = {x * voltageComp, y * voltageComp, h * voltageComp};
        Vector2d vec = new Vector2d(p[0], p[1]).rotated(-currHeading);
        return new Pose2d(vec.getX(), vec.getY(), p[2]); //clockwise
    }

    private List<Double> getDrivePower(Pose2d pose2d) {
        return MecanumKinematics.robotToWheelVelocities(pose2d, 1, 1, lateralMult);
    }

}
