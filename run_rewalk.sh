export PYTHONPATH=$PYTHONPATH:/Path/to/GUI-ReWalk
export API_KEY="your model api key"
export API_BASE_URL="your model api base url"


python gui_rewalk/run_random_walker.py \
    --task_num 1 \
    --vm_provider "vmware" \
    --path_to_vm  "Path/to/Ubuntu0.vmx" \
    --observation_type screenshot \
    --action_space gen_data \
    --model doubao or qwen \
    --model_version your model api name\
    --exce_task_completion True \
    --reverse_inference True \
    --summary_inference True \
    --max_random_actions 2 \
    --max_guided_actions 2 \
    --random_walk_cross_app 1 \
    --max_guided_actions_after_openapp 3 \
    --max_trajectory_length 1 \
    --score_threshold 2 \
    --pq_format qwen_train_multi_steps \
    --ocr_model_path OmniParser/weights/icon_detect/model.pt \
    --random_walker True \
    --enable_thinking True \
    --use_ark True \
    --result_dir result
