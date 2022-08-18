from frankx import Robot, JointMotion


class RobotController():
    def __init__(self, ip, dynamic_limit_rel=0.1, joint_change_limit_rad=0.05) -> None:
        self.ip = ip
        self.robot = Robot(ip)
        # Percentage of robot's maximum velocity, acceleration and jerk
        self.robot.set_dynamic_rel(dynamic_limit_rel)
        self.joint_change_limit_rad = joint_change_limit_rad

    def limit_joint_motion(self, curr, new):
        if new < curr - self.joint_change_limit_rad:
            return curr - self.joint_change_limit_rad
        if new > curr + self.joint_change_limit_rad:
            return curr + self.joint_change_limit_rad
        return new

    def move(self, pitch, roll, relative=False):
        state = self.robot.get_state()
        if any(dq > 0.1 for dq in state.dq):
            print('Warning: robot is currently moving!')

        j = state.q
        if relative:
            pitch += j[3]
            roll += j[4]
        j[3] = self.limit_joint_motion(j[3], pitch)  # elbow
        j[4] = self.limit_joint_motion(j[4], roll)  # forearm

        try:
            self.robot.recover_from_errors()
            self.robot.move(JointMotion(j))
        except Exception as e:
            print(str(e))

    def move_rel(self, dpitch, droll):

