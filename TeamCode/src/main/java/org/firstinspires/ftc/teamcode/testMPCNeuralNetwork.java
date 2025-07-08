package org.firstinspires.ftc.teamcode;

import static org.firstinspires.ftc.teamcode.trajectorysequence.TrajectorySequenceRunner.COLOR_ACTIVE_TRAJECTORY;
import static org.firstinspires.ftc.teamcode.trajectorysequence.TrajectorySequenceRunner.COLOR_INACTIVE_TRAJECTORY;

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
    DcMotorEx leftFront = hardwareMap.get(DcMotorEx.class, "leftFront");
    DcMotorEx leftRear = hardwareMap.get(DcMotorEx.class, "leftRear");
    DcMotorEx rightRear = hardwareMap.get(DcMotorEx.class, "rightRear");
    DcMotorEx rightFront = hardwareMap.get(DcMotorEx.class, "rightFront");
    private ElapsedTime timer = new ElapsedTime();
    private FtcDashboard dashboard;


    @Override
    public void runOpMode() throws InterruptedException {
        Telemetry telemetry = new MultipleTelemetry(this.telemetry, FtcDashboard.getInstance().getTelemetry());
//        Session session = null;
//        String modelPath = "\\OneDrive\\Documents\\MATLAB\\nlmpcImitate\\savedModel";
//        try (SavedModelBundle model = SavedModelBundle.load(modelPath, "serve")) {
//            session = model.session();
//        } catch (Exception e) {
//
//            e.printStackTrace();
//        }
        Interpreter interpreter1;
        try (Interpreter interpreter2 = new Interpreter(new File("model.tflite"))) {
            interpreter1 = interpreter2;
        }
        double[] inputs = {1,2,3,4,5,6,7};
        double[] outputs = {1,2,3,4};

        Pose2d pose;
        double heading;
        imu = hardwareMap.get(IMU.class, "imu");
        imu.initialize(new IMU.Parameters(new RevHubOrientationOnRobot(
                RevHubOrientationOnRobot.LogoFacingDirection.UP,
                RevHubOrientationOnRobot.UsbFacingDirection.FORWARD
        )));
        localizer = new TwoWheelTrackingLocalizer(leftFront, leftRear,
                () -> AngleUnit.normalizeRadians(imu.getRobotYawPitchRollAngles().getYaw(AngleUnit.RADIANS)),
                () -> imu.getRobotAngularVelocity(AngleUnit.RADIANS).zRotationRate); //parallel, perp, orientation, velocity
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
            heading = getHeading();
            inputs[0] = pose.getX();
            inputs[1] = pose.getY();
            inputs[2] = heading;
            inputs[3] = goal1[0];
            inputs[4] = goal1[1];
            inputs[5] = goal2[0];
            inputs[6] = goal2[1];


            timer.reset();
            double startTime = timer.milliseconds();
            interpreter1.run(inputs, outputs);
            double totalTime = timer.milliseconds() - startTime;


            telemetry.addData("totalTime", totalTime);
            telemetry.addData("outputs", outputs);
            telemetry.addData("currentPos", new double[]{inputs[0], inputs[1], inputs[2]});
            telemetry.addData("goal1", goal1);
            telemetry.addData("goal2", goal2);
            DashboardUtil.drawRobot(fieldOverlay, pose);
            fieldOverlay.setStroke(COLOR_INACTIVE_TRAJECTORY);
            DashboardUtil.drawSampledPath(fieldOverlay, new Path(new PathSegment(
                    new LineSegment(new Vector2d(inputs[0], inputs[1]),
                            new Vector2d(inputs[3], inputs[4])))));
            DashboardUtil.drawSampledPath(fieldOverlay, new Path(new PathSegment(
                    new LineSegment(new Vector2d(inputs[3], inputs[4]),
                            new Vector2d(inputs[5], inputs[6])))));
            telemetry.update();
        }
    }
    double getHeading() {
        return imu.getRobotYawPitchRollAngles().getYaw(AngleUnit.RADIANS);
    }


}
