"""
LLM 姿态时序纠错 Agent
基于自注意力机制的骨骼关键点时序漂移纠错
模拟 Llama V2 Pose 边缘端推理，使用本地 Ollama (qwen2.5:7b) 作为推理后端

核心功能：
1. 分析连续帧的关键点轨迹，检测异常漂移
2. 基于自注意力机制评估关键点间的时空一致性
3. 对跌倒/在床判别进行二次确认与纠错
4. 生成自然语言分析报告
"""
import json
import logging
import numpy as np
from typing import Optional, List, Dict, Tuple
from collections import deque
import urllib.request
import urllib.error

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import LLM_AGENT, COCO_KEYPOINTS, BehaviorState
from poses.yolo_pose import PersonPose

logger = logging.getLogger(__name__)


class PoseCorrector:
    """
    姿态时序纠错器
    模拟 Llama V2 Pose 的自注意力骨骼关键点时序漂移纠错机制
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = {**LLM_AGENT, **(config or {})}
        self.enabled = self.config["enabled"]
        self.base_url = self.config["base_url"]
        self.model = self.config["model"]
        self.temporal_window = self.config["temporal_window"]

        # 每个人的关键点历史
        self.keypoint_history: Dict[int, deque] = {}
        # 每个人的状态历史
        self.state_history: Dict[int, deque] = {}
        # Ollama 可用性
        self._ollama_available = None

    def check_ollama(self) -> bool:
        """检查 Ollama 服务是否可用"""
        if self._ollama_available is not None:
            return self._ollama_available

        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read())
                models = [m["name"] for m in data.get("models", [])]
                self._ollama_available = self.model in models or any(
                    self.model.split(":")[0] in m for m in models
                )
                logger.info(f"Ollama可用，模型列表: {models}")
        except Exception as e:
            logger.warning(f"Ollama不可用: {e}，将使用纯算法纠错")
            self._ollama_available = False

        return self._ollama_available

    def _call_ollama(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        """调用 Ollama API"""
        if not self.check_ollama():
            return None

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 512,
            }
        }
        if system_prompt:
            payload["system"] = system_prompt

        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                f"{self.base_url}/api/generate",
                data=data,
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=self.config["timeout"]) as resp:
                result = json.loads(resp.read())
                return result.get("response", "")
        except Exception as e:
            logger.warning(f"Ollama调用失败: {e}")
            return None

    def update_history(self, pose: PersonPose, state: str):
        """更新历史记录"""
        track_id = pose.track_id
        if track_id not in self.keypoint_history:
            self.keypoint_history[track_id] = deque(maxlen=self.temporal_window)
            self.state_history[track_id] = deque(maxlen=self.temporal_window)

        self.keypoint_history[track_id].append(pose.keypoints.copy())
        self.state_history[track_id].append(state)

    def correct_classification(self, pose: PersonPose, state: str,
                                 details: Dict) -> Tuple[str, float, str]:
        """
        对分类结果进行LLM二次确认与纠错

        Returns:
            (corrected_state, confidence, explanation)
        """
        self.update_history(pose, state)

        track_id = pose.track_id
        kp_history = self.keypoint_history.get(track_id, deque())
        state_hist = self.state_history.get(track_id, deque())

        # 纯算法纠错（始终执行）
        algo_corrected, algo_conf, algo_reason = self._algorithm_correction(
            pose, state, details, kp_history, state_hist
        )

        # LLM 纠错（如果可用且需要）
        # 仅在跌倒状态或明确的状态冲突时调用LLM，避免频繁调用导致性能问题
        if self.enabled and self.check_ollama() and len(kp_history) >= 5:
            should_call_llm = False
            if state == BehaviorState.FALLEN:
                should_call_llm = True  # 跌倒状态必须二次确认
            elif state != algo_corrected and algo_corrected == BehaviorState.FALLEN:
                should_call_llm = True  # 算法纠错为跌倒，需确认
            elif len(set(list(state_hist)[-5:])) > 1 and BehaviorState.FALLEN in state_hist:
                should_call_llm = True  # 状态波动且包含跌倒，需确认

            if should_call_llm:
                llm_result = self._llm_correction(pose, state, details,
                                                     kp_history, state_hist)
                if llm_result is not None:
                    return llm_result

        return algo_corrected, algo_conf, algo_reason

    def _algorithm_correction(self, pose: PersonPose, state: str,
                                details: Dict, kp_history: deque,
                                state_hist: deque) -> Tuple[str, float, str]:
        """
        纯算法时序纠错
        基于自注意力的关键点时空一致性评估
        """
        if len(kp_history) < 3:
            return state, 0.7, "历史帧不足，跳过纠错"

        # 1. 检测关键点漂移
        drift_score = self._compute_drift_score(kp_history)

        # 2. 检测状态一致性
        state_consistency = self._compute_state_consistency(state_hist)

        # 3. 跌倒事件检测（速度+角度突变）
        fall_event = self._detect_fall_event(kp_history)

        # 纠错逻辑
        corrected = state
        confidence = 0.7
        reason = "算法纠错通过"

        if state == BehaviorState.FALLEN:
            if fall_event and state_consistency > 0.5:
                confidence = 0.95
                reason = f"跌倒事件确认: 速度突变+水平姿态, 状态一致性{state_consistency:.0%}"
            elif not fall_event:
                # 可能是误报（正常躺卧被误判为跌倒）
                if details.get("iou", 0) > 0.2 or details.get("keypoint_in_bed_ratio", 0) > 0.4:
                    corrected = BehaviorState.ON_BED
                    confidence = 0.75
                    reason = "纠错: 人体与床体重叠度较高，修正为在床"
                else:
                    confidence = 0.6
                    reason = "跌倒待确认: 未检测到典型跌倒速度突变"

        elif state == BehaviorState.ON_BED:
            if drift_score > 0.7 and fall_event:
                # 可能是从床上摔落
                corrected = BehaviorState.FALLEN
                confidence = 0.8
                reason = "纠错: 检测到从床体区域快速移出+姿态变水平，修正为跌倒"

        return corrected, confidence, reason

    def _compute_drift_score(self, kp_history: deque) -> float:
        """
        计算关键点漂移分数（自注意力机制简化版）
        评估相邻帧之间关键点的位移一致性
        """
        if len(kp_history) < 2:
            return 0.0

        displacements = []
        for i in range(1, len(kp_history)):
            prev = kp_history[i - 1][:, :2]
            curr = kp_history[i][:, :2]
            valid = kp_history[i][:, 2] > 0.3
            if valid.sum() > 0:
                disp = np.linalg.norm(curr[valid] - prev[valid], axis=1)
                displacements.append(np.mean(disp))

        if len(displacements) < 2:
            return 0.0

        # 位移突变比（当前帧位移 / 平均位移）
        recent = displacements[-1]
        avg = np.mean(displacements[:-1]) if len(displacements) > 1 else recent
        if avg < 1e-6:
            return 0.0

        ratio = recent / avg
        return float(min(ratio / 5.0, 1.0))  # 归一化到[0,1]

    def _compute_state_consistency(self, state_hist: deque) -> float:
        """计算状态一致性（最近N帧中同一状态的比例）"""
        if len(state_hist) < 2:
            return 1.0
        recent = list(state_hist)[-5:]
        from collections import Counter
        most_common_count = Counter(recent).most_common(1)[0][1]
        return most_common_count / len(recent)

    def _detect_fall_event(self, kp_history: deque) -> bool:
        """
        检测跌倒事件
        特征：臀部关键点y坐标快速增大（向下运动）+ 人体轴线从垂直变水平
        """
        if len(kp_history) < 5:
            return False

        # 臀部y坐标变化率
        hip_indices = [11, 12]  # left_hip, right_hip
        hip_ys = []
        for kps in kp_history:
            valid_hips = [kps[i, 1] for i in hip_indices if kps[i, 2] > 0.3]
            if valid_hips:
                hip_ys.append(np.mean(valid_hips))
            else:
                hip_ys.append(None)

        valid_ys = [y for y in hip_ys if y is not None]
        if len(valid_ys) < 4:
            return False

        # 最近几帧的y变化
        recent_ys = valid_ys[-4:]
        y_change = recent_ys[-1] - recent_ys[0]

        # 人体轴线角度变化
        angles = []
        for kps in list(kp_history)[-5:]:
            left_shoulder = kps[5]
            right_shoulder = kps[6]
            left_hip = kps[11]
            right_hip = kps[12]
            if all(kp[2] > 0.3 for kp in [left_shoulder, right_shoulder, left_hip, right_hip]):
                shoulder_c = np.mean([left_shoulder[:2], right_shoulder[:2]], axis=0)
                hip_c = np.mean([left_hip[:2], right_hip[:2]], axis=0)
                dx = hip_c[0] - shoulder_c[0]
                dy = hip_c[1] - shoulder_c[1]
                angle = np.degrees(np.arctan2(abs(dy), abs(dx)))
                angles.append(angle)

        angle_change = 0
        if len(angles) >= 3:
            angle_change = angles[0] - angles[-1]  # 从大角度（垂直）变小角度（水平）

        # 跌倒判定：y快速下降（向下）+ 角度从垂直变水平
        return y_change > 30 and angle_change > 20

    def _llm_correction(self, pose: PersonPose, state: str, details: Dict,
                          kp_history: deque, state_hist: deque) -> Optional[Tuple[str, float, str]]:
        """
        LLM 纠错（模拟 Llama V2 Pose 的自注意力推理）
        将关键点时序数据和状态历史交给LLM分析
        """
        # 构建关键点轨迹摘要
        trajectory_summary = self._build_trajectory_summary(kp_history)
        state_summary = list(state_hist)[-10:]

        system_prompt = """你是一个专业的人体姿态分析专家，类似于 Llama V2 Pose 模型。
你的任务是分析人体骨骼关键点的时序轨迹，判断当前行为状态是否准确。
重点关注：
1. 关键点的运动轨迹是否符合跌倒/在床/活动的物理规律
2. 自注意力机制：各关键点之间的时空一致性
3. 跌倒的典型特征：快速向下运动 + 身体从垂直变为水平
4. 在床的典型特征：身体水平 + 与床体区域重叠 + 运动缓慢
请以JSON格式回复：{"state": "on_bed/fallen/active/unknown", "confidence": 0.0-1.0, "reason": "分析原因"}"""

        prompt = f"""当前状态: {state}
当前帧详情: IoU={details.get('iou', 0):.3f}, 关键点在床比例={details.get('keypoint_in_bed_ratio', 0):.3f}, 人体轴线角度={details.get('body_axis_angle', 0):.1f}°, 臀部y={details.get('hip_y', 0):.0f}, 地面边缘y={details.get('floor_edge_y', 0)}

最近状态序列: {state_summary}

关键点轨迹摘要（最近{len(trajectory_summary)}帧）:
{json.dumps(trajectory_summary, ensure_ascii=False, indent=2)}

请分析当前状态判定是否准确，并给出纠错建议。"""

        response = self._call_ollama(prompt, system_prompt)
        if response is None:
            return None

        # 解析LLM回复
        try:
            # 提取JSON
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                result = json.loads(response[json_start:json_end])
                llm_state = result.get("state", state)
                llm_conf = float(result.get("confidence", 0.7))
                llm_reason = result.get("reason", "LLM分析")

                # 状态映射
                state_map = {
                    "on_bed": BehaviorState.ON_BED,
                    "fallen": BehaviorState.FALLEN,
                    "active": BehaviorState.ACTIVE,
                    "unknown": BehaviorState.UNKNOWN,
                }
                corrected = state_map.get(llm_state.lower(), state)
                return corrected, llm_conf, f"[LLM] {llm_reason}"
        except Exception as e:
            logger.warning(f"LLM回复解析失败: {e}")

        return None

    def _build_trajectory_summary(self, kp_history: deque) -> List[Dict]:
        """构建关键点轨迹摘要（降采样，减少token）"""
        summary = []
        # 每2帧取一帧，最多取8帧
        indices = list(range(0, len(kp_history), max(1, len(kp_history) // 8)))
        indices = indices[-8:]  # 最多8帧

        for idx in indices:
            kps = kp_history[idx]
            frame_data = {}
            # 只记录重点关键点
            key_names = ["left_shoulder", "right_shoulder", "left_hip", "right_hip",
                         "left_knee", "right_knee", "left_ankle", "right_ankle"]
            for name in key_names:
                kp_idx = COCO_KEYPOINTS.index(name)
                kp = kps[kp_idx]
                if kp[2] > 0.3:
                    frame_data[name] = [round(float(kp[0]), 1), round(float(kp[1]), 1)]
            summary.append(frame_data)

        return summary

    def generate_report(self, track_id: int) -> str:
        """生成某个人的行为分析报告"""
        if track_id not in self.state_history:
            return "无历史数据"

        states = list(self.state_history[track_id])
        from collections import Counter
        state_counts = Counter(states)

        report = f"=== 行为分析报告 (Track ID: {track_id}) ===\n"
        report += f"总分析帧数: {len(states)}\n"
        for state, count in state_counts.most_common():
            report += f"  {state}: {count}帧 ({count/len(states)*100:.1f}%)\n"

        # 跌倒事件统计
        fall_frames = [i for i, s in enumerate(states) if s == BehaviorState.FALLEN]
        if fall_frames:
            report += f"\n跌倒事件: 检测到 {len(fall_frames)} 帧跌倒\n"
            # 连续跌倒段
            segments = []
            start = fall_frames[0]
            for i in range(1, len(fall_frames)):
                if fall_frames[i] - fall_frames[i - 1] > 5:
                    segments.append((start, fall_frames[i - 1]))
                    start = fall_frames[i]
            segments.append((start, fall_frames[-1]))
            for i, (s, e) in enumerate(segments):
                report += f"  跌倒段 {i + 1}: 第{s}帧 - 第{e}帧 (持续{e - s + 1}帧)\n"

        return report

    def reset(self):
        """重置纠错器状态"""
        self.keypoint_history.clear()
        self.state_history.clear()
