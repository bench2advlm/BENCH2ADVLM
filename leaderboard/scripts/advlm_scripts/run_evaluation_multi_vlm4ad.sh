#!/bin/bash
BASE_PORT=30003
BASE_TM_PORT=50003
IS_BENCH2DRIVE=True
BASE_ROUTES=leaderboard/data/bench2drive220
TEAM_AGENT=team_code/vlm4ad_b2d_agent.py
# [llama/llava]+[generation/selection]
TEAM_CONFIG=llama+generation
ALGO=vlm4ad+llama+generation
BASE_CHECKPOINT_ENDPOINT=eval_bench2drive220
PLANNER_TYPE=traj
SAVE_PATH=./eval_bench2drive220_${ALGO}_${PLANNER_TYPE}

if [ ! -d "${ALGO}_b2d_${PLANNER_TYPE}" ]; then
    mkdir ${ALGO}_b2d_${PLANNER_TYPE}
    echo -e "\033[32m Directory ${ALGO}_b2d_${PLANNER_TYPE} created. \033[0m"
else
    echo -e "\033[32m Directory ${ALGO}_b2d_${PLANNER_TYPE} already exists. \033[0m"
fi

# Check if the split_xml script needs to be executed
if [ ! -f "${BASE_ROUTES}_${ALGO}_${PLANNER_TYPE}_split_done.flag" ]; then
    echo -e "****************************\033[33m Attention \033[0m ****************************"
    echo -e "\033[33m Running split_xml.py \033[0m"
    TASK_NUM=1
    python tools/split_xml.py $BASE_ROUTES $TASK_NUM $ALGO $PLANNER_TYPE
    touch "${BASE_ROUTES}_${ALGO}_${PLANNER_TYPE}_split_done.flag"
    echo -e "\033[32m Splitting complete. Flag file created. \033[0m"
else
    echo -e "\033[32m Splitting already done. \033[0m"
fi

echo -e "**************\033[36m Please Manually adjust GPU or TASK_ID \033[0m **************"
GPU_RANK_LIST=(3)
TASK_LIST=(0)
echo -e "\033[32m GPU_RANK_LIST: $GPU_RANK_LIST \033[0m"
echo -e "\033[32m TASK_LIST: $TASK_LIST \033[0m"
echo -e "***********************************************************************************"

length=${#GPU_RANK_LIST[@]}
for ((i=0; i<$length; i++ )); do
    PORT=$((BASE_PORT + i * 150))
    TM_PORT=$((BASE_TM_PORT + i * 150))
    ROUTES="${BASE_ROUTES}_${TASK_LIST[$i]}_${ALGO}_${PLANNER_TYPE}.xml"
    CHECKPOINT_ENDPOINT="${ALGO}_b2d_${PLANNER_TYPE}/${BASE_CHECKPOINT_ENDPOINT}_${TASK_LIST[$i]}.json"
    GPU_RANK=${GPU_RANK_LIST[$i]}
    echo -e "\033[32m ALGO: $ALGO \033[0m"
    echo -e "\033[32m PLANNER_TYPE: $PLANNER_TYPE \033[0m"
    echo -e "\033[32m TASK_ID: $i \033[0m"
    echo -e "\033[32m PORT: $PORT \033[0m"
    echo -e "\033[32m TM_PORT: $TM_PORT \033[0m"
    echo -e "\033[32m CHECKPOINT_ENDPOINT: $CHECKPOINT_ENDPOINT \033[0m"
    echo -e "\033[32m GPU_RANK: $GPU_RANK \033[0m"
    echo -e "\033[32m bash leaderboard/scripts/advlm_scripts/run_evaluation_advlm.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK \033[0m"
    echo -e "***********************************************************************************"
    bash -e leaderboard/scripts/advlm_scripts/run_evaluation_advlm.sh $PORT $TM_PORT $IS_BENCH2DRIVE $ROUTES $TEAM_AGENT $TEAM_CONFIG $CHECKPOINT_ENDPOINT $SAVE_PATH $PLANNER_TYPE $GPU_RANK 2>&1 > ${BASE_ROUTES}_${TASK_LIST[$i]}_${ALGO}_${PLANNER_TYPE}.log &
    sleep 5
done
wait