package org.firstinspires.ftc.teamcode;

import static org.firstinspires.ftc.teamcode.trajectorysequence.TrajectorySequenceRunner.COLOR_ACTIVE_TRAJECTORY;
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
import com.qualcomm.robotcore.eventloop.opmode.OpMode;
import com.qualcomm.robotcore.hardware.DcMotor;
import com.qualcomm.robotcore.hardware.DcMotorEx;
import com.qualcomm.robotcore.hardware.IMU;
import com.qualcomm.robotcore.util.ElapsedTime;
//import org.tensorflow.SavedModelBundle;
//import org.tensorflow.Session;
//import org.tensorflow.Tensor;
import org.firstinspires.ftc.robotcore.external.Telemetry;
import org.firstinspires.ftc.teamcode.util.DashboardUtil;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.TensorFlowLite;



import org.firstinspires.ftc.robotcore.external.navigation.AngleUnit;
import org.firstinspires.ftc.robotcore.external.navigation.Pose2D;
import org.firstinspires.ftc.robotcore.external.navigation.Pose3D;
import org.firstinspires.ftc.teamcode.drive.TwoWheelTrackingLocalizer;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Paths;
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
        float[] inputs = {1,2,3,4,5,6,7};
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
        TelemetryPacket initialPacket = new TelemetryPacket();
        Canvas fieldOverlay = initialPacket.fieldOverlay();
        dashboard.sendTelemetryPacket(initialPacket);
        DashboardUtil.drawRobot(fieldOverlay, new Pose2d(0,0));


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

            double[] goal1 = {10, 10};
            double[] goal2 = {20,20};
            pose = localizer.getPoseEstimate();
            inputs[0] = (float) pose.getX();
            inputs[1] = (float) pose.getY();
            inputs[2] = (float) pose.getHeading();
            inputs[3] = (float) goal1[0];
            inputs[4] = (float) goal1[1];
            inputs[5] = (float) goal2[0];
            inputs[6] = (float) goal2[1];


            timer.reset();
            double startTime = timer.milliseconds();
            interpreter.run(inputs, outputs);
            double totalTime = timer.milliseconds() - startTime;

            // [fr, fl, br, bl]
//            fr.setVelocity();


            telemetry.addData("totalTime", totalTime);
            telemetry.addData("outputs", Arrays.toString(outputs[0]));
            telemetry.addData("currentPos", Arrays.toString(new double[]{inputs[0], inputs[1], inputs[2]}));
            telemetry.addData("goal1", Arrays.toString(goal1));
            telemetry.addData("goal2", Arrays.toString(goal2));
            telemetry.addData("parallel", fl.getCurrentPosition());
            telemetry.addData("perp", bl.getCurrentPosition());

            TelemetryPacket packet = new TelemetryPacket();
            fieldOverlay = packet.fieldOverlay();
            DashboardUtil.drawRobot(fieldOverlay, pose);
            fieldOverlay.setStroke(COLOR_INACTIVE_TRAJECTORY);
            DashboardUtil.drawSampledPath(fieldOverlay, new Path(new PathSegment(
                    new LineSegment(new Vector2d(inputs[0], inputs[1]),
                            new Vector2d(inputs[3], inputs[4])))));
            DashboardUtil.drawSampledPath(fieldOverlay, new Path(new PathSegment(
                    new LineSegment(new Vector2d(inputs[3], inputs[4]),
                            new Vector2d(inputs[5], inputs[6])))));
            dashboard.sendTelemetryPacket(packet);
            telemetry.update();
            localizer.update();
        }
    }
    double getHeading() {
        return imu.getRobotYawPitchRollAngles().getYaw(AngleUnit.RADIANS);
    }
    private MappedByteBuffer loadModelFile() throws IOException {
        String modelPath = "modelSmaller.tflite";
        AssetFileDescriptor fileDescriptor =
                hardwareMap.appContext.getAssets().openFd(modelPath);
        FileInputStream inputStream =
                new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
}
