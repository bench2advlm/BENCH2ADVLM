#!/bin/bash
export CARLA_ROOT=/home/beihang/zty/jt/carla
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:leaderboard
export PYTHONPATH=$PYTHONPATH:leaderboard/team_code
export PYTHONPATH=$PYTHONPATH:scenario_runner
export PYTHONPATH=$PYTHONPATH:ADVLM
export PYTHONPATH=$PYTHONPATH:GVLM
export PYTHONPATH=$PYTHONPATH:GVLM/llama3
export PYTHONPATH=$PYTHONPATH:GVLM/LLaVA

export SCENARIO_RUNNER_ROOT=scenario_runner
export LEADERBOARD_ROOT=leaderboard
export CHALLENGE_TRACK_CODENAME=SENSORS
export PORT=$1
export TM_PORT=$2
export DEBUG_CHALLENGE=0
export REPETITIONS=1
export RESUME=True
export IS_BENCH2DRIVE=$3
export ROUTES=$4
export TEAM_AGENT=$5
export TEAM_CONFIG=$6
export CHECKPOINT_ENDPOINT=$7
export SAVE_PATH=$8
export PLANNER_TYPE=$9
export GPU_RANK=${10}
export PARSER_TYPE="${TEAM_CONFIG%%+*}"
export SIGNAL_TYPE="${TEAM_CONFIG#*+}"

if [[ "$PARSER_TYPE" == "llama" ]]; then
    MASTER_PORT=$((29400 + GPU_RANK))
    CUDA_VISIBLE_DEVICES=${GPU_RANK} torchrun --nproc_per_node 1 --master_port=${MASTER_PORT} ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_vlm_llama.py \
	--routes=${ROUTES} \
	--repetitions=${REPETITIONS} \
	--track=${CHALLENGE_TRACK_CODENAME} \
	--checkpoint=${CHECKPOINT_ENDPOINT} \
	--agent=${TEAM_AGENT} \
	--agent-config=${TEAM_CONFIG} \
	--debug=${DEBUG_CHALLENGE} \
	--record=${RECORD_PATH} \
	--resume=${RESUME} \
	--port=${PORT} \
	--traffic-manager-port=${TM_PORT} \
	--gpu-rank=${GPU_RANK}
else
    CUDA_VISIBLE_DEVICES=${GPU_RANK} python ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator_vlm_llava.py \
     --routes=${ROUTES} \
	--repetitions=${REPETITIONS} \
	--track=${CHALLENGE_TRACK_CODENAME} \
	--checkpoint=${CHECKPOINT_ENDPOINT} \
	--agent=${TEAM_AGENT} \
	--agent-config=${TEAM_CONFIG} \
	--debug=${DEBUG_CHALLENGE} \
	--record=${RECORD_PATH} \
	--resume=${RESUME} \
	--port=${PORT} \
	--traffic-manager-port=${TM_PORT} \
	--gpu-rank=${GPU_RANK}
fi