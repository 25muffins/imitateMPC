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
//import org.tensorflow.SavedModelBundle;
//import org.tensorflow.Session;
//import org.tensorflow.Tensor;
import org.firstinspires.ftc.robotcore.external.Telemetry;
import org.firstinspires.ftc.teamcode.util.DashboardUtil;
import org.tensorflow.lite.Interpreter;


import org.firstinspires.ftc.robotcore.external.navigation.AngleUnit;
import org.firstinspires.ftc.teamcode.drive.TwoWheelTrackingLocalizer;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Arrays;
//import org.tensorflow.ndarray.IntNdArray;
//import org.tensorflow.ndarray.NdArrays;
//import org.tensorflow.ndarray.Shape;
//import org.tensorflow.types.TFloat32;
//import org.tensorflow.types.TInt32;

@Config
@Autonomous(name = "testMPCNeuralNetwork")
public class testMPCNeuralNetwork extends LinearOpMode {
    IMU imu;
    TwoWheelTrackingLocalizer localizer;

    private ElapsedTime timer = new ElapsedTime();
    private FtcDashboard dashboard;
//for testing directions
//    public static double flSpeed = 0;
//    public static double blSpeed = 0;
//    public static double brSpeed = 0;
//    public static double frSpeed = 0;

    public static double speed = 0.5;
    public static double[] goal1 = {10, -10};
    public static double[] goal2 = {20,-20};


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


//        Session session = null;
//        String modelPath = "\\OneDrive\\Documents\\MATLAB\\nlmpcImitate\\savedModel";
//        try (SavedModelBundle model = SavedModelBundle.load(modelPath, "serve")) {
//            session = model.session();
//        } catch (Exception e) {
//
//            e.printStackTrace();
//        }
        Interpreter interpreter;
        try  {
            interpreter = new Interpreter(loadModelFile());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
        float[] inputs = {0,0,0, (float) goal1[0], (float) goal1[1], (float) goal2[0], (float) goal2[1]};
        float[][] outputs = {{1,2,3,4}};

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
        drawField(new Pose2d(0,0,0), inputs);


        waitForStart();
        while(opModeIsActive()){


//            IntNdArray input_matrix = NdArrays.ofInts(Shape.of(1, 9));
//            input_matrix.set(NdArrays.vectorOf(1, 2, 3, 5, 7, 21, 23, 43, 123), 0);
//            Tensor input_tensor = TFloat32.tensorOf(input_matrix.shape());
//
//            Tensor outputTensor = session.runner()
//                    .feed(inputTensorName, inputTensor)
//                    .fetch(outputTensorName)
//                    .run()
//                    .get(0); // Get the first (and likely only) output tensor
//            outputTensor


            pose = localizer.getPoseEstimate();
            inputs[0] = (float) pose.getX();
            inputs[1] = (float) pose.getY();

            float HeadingChange = (float) (pose.getHeading() - Math.PI/4); //idk why i think it has something to do with the switched x and y
            if(HeadingChange > Math.PI) { inputs[2] = (float) (HeadingChange - 2*Math.PI);}
            else{ inputs[2] = HeadingChange;}

            inputs[3] = (float) goal1[0];
            inputs[4] = (float) -goal1[1];
            inputs[5] = (float) goal2[0];
            inputs[6] = (float) -goal2[1];


            timer.reset();
            double startTime = timer.milliseconds();
            interpreter.run(inputs, outputs);
            double totalTime = timer.milliseconds() - startTime;

            // [fr, fl, br, bl]
//            fr.setVelocity();
            fr.setPower(speed*outputs[0][0]);
            fl.setPower(speed*outputs[0][1]);
            br.setPower(speed*outputs[0][2]);
            bl.setPower(speed*outputs[0][3]);
            //for testing directions
//            fl.setPower(flSpeed);
//            bl.setPower(blSpeed);
//            br.setPower(brSpeed);
//            fr.setPower(frSpeed);

            telemetry.addData("totalTime", totalTime);
            telemetry.addData("outputs", Arrays.toString(outputs[0]));
            telemetry.addData("currentPos", Arrays.toString(new double[]{inputs[0], inputs[1], inputs[2]}));
            telemetry.addData("goal1", Arrays.toString(goal1));
            telemetry.addData("goal2", Arrays.toString(goal2));
            telemetry.addData("parallel", fl.getCurrentPosition());
            telemetry.addData("perp", bl.getCurrentPosition());
            telemetry.addData("imu", getHeading());


            drawField(pose, inputs);

            telemetry.update();
            localizer.update();
        }
    }
    double getHeading() {
        return imu.getRobotYawPitchRollAngles().getYaw(AngleUnit.RADIANS);
    }
    private MappedByteBuffer loadModelFile() throws IOException {
        String modelPath = "model.tflite";
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
                new LineSegment(new Vector2d(inputs[3], inputs[4]),
                        new Vector2d(goal2[0], goal2[1])))));
        dashboard.sendTelemetryPacket(packet);
    }

}
