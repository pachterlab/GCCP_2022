#hidden units, do some search over different learning rates and batchsizes
num_hidden_units=("64" "128" "256" "512") 
num_hidden_layers=("1" "3" "4" "5") 


for hl in "${num_hidden_layers[@]}"
do

#   python3 trainVaryLoss.py --batchsize 100 --num_units 256 --num_epochs 30 --num_training_files 5 --npdf $1 --lr 1e-3 --num_training_params 256 --num_layers 2 --scale_or_hyp "hyp" --loss_function "mse"
#   python3 trainVaryLoss.py --batchsize 100 --num_units ${hu} --num_epochs 30 --num_training_files 5 --npdf $1 --lr 1e-3 --num_training_params 256 --num_layers 2 --scale_or_hyp "hyp" --loss_function "hellinger"
  
  
#   python3 trainVaryLoss.py --batchsize 100 --num_units 256 --num_epochs 30 --num_training_files 5 --npdf $1 --lr 1e-3 --num_training_params 256 --num_layers 2 --scale_or_hyp "scale" --max_mv "no_update" --loss_function "mse"
#   python3 trainVaryLoss.py --batchsize 100 --num_units ${hu} --num_epochs 30 --num_training_files 5 --npdf $1 --lr 1e-3 --num_training_params 256 --num_layers 2 --scale_or_hyp "scale" --max_mv "no_update" --loss_function "hellinger"
  
#   python3 trainVaryLoss.py --batchsize 100 --num_units 256 --num_epochs 30 --num_training_files 5 --npdf $1 --lr 1e-3 --num_training_params 256 --num_layers ${hl} --scale_or_hyp "scale" --max_mv "update" --loss_function "mse"
  python3 trainVaryLoss.py --batchsize 100 --num_units 256 --num_epochs 30 --num_training_files 5 --npdf 3 --lr 1e-3 --num_training_params 256 --num_layers ${hl} --scale_or_hyp "scale" --max_mv "update" --loss_function "joint_hellinger"
  python3 trainVaryLoss.py --batchsize 100 --num_units 256 --num_epochs 30 --num_training_files 5 --npdf 5 --lr 1e-3 --num_training_params 256 --num_layers ${hl} --scale_or_hyp "scale" --max_mv "update" --loss_function "joint_hellinger"
  python3 trainVaryLoss.py --batchsize 100 --num_units 256 --num_epochs 30 --num_training_files 5 --npdf 10 --lr 1e-3 --num_training_params 256 --num_layers ${hl} --scale_or_hyp "scale" --max_mv "update" --loss_function "joint_hellinger"

done

for hu in "${num_hidden_units[@]}"
do

#   python3 trainVaryLoss.py --batchsize 100 --num_units ${hu} --num_epochs 50 --num_training_files 5 --npdf $1 --lr 1e-3 --num_training_params 256 --num_layers 2 --scale_or_hyp "hyp" --loss_function "mse"
#   python3 trainVaryLoss.py --batchsize 100 --num_units ${hu} --num_epochs 50 --num_training_files 5 --npdf $1 --lr 1e-3 --num_training_params 256 --num_layers 2 --scale_or_hyp "hyp" --loss_function "hellinger"
  
  
#   python3 trainVaryLoss.py --batchsize 100 --num_units ${hu} --num_epochs 50 --num_training_files 5 --npdf $1 --lr 1e-3 --num_training_params 256 --num_layers 2 --scale_or_hyp "scale" --max_mv "no_update" --loss_function "mse"
#   python3 trainVaryLoss.py --batchsize 100 --num_units ${hu} --num_epochs 50 --num_training_files 5 --npdf $1 --lr 1e-3 --num_training_params 256 --num_layers 2 --scale_or_hyp "scale" --max_mv "no_update" --loss_function "hellinger"
  
#   python3 trainVaryLoss.py --batchsize 100 --num_units ${hu} --num_epochs 50 --num_training_files 5 --npdf $1 --lr 1e-3 --num_training_params 256 --num_layers 2 --scale_or_hyp "scale" --max_mv "update" --loss_function "mse"
  python3 trainVaryLoss.py --batchsize 100 --num_units ${hu} --num_epochs 50 --num_training_files 5 --npdf 3 --lr 1e-3 --num_training_params 256 --num_layers 2 --scale_or_hyp "scale" --max_mv "update" --loss_function "joint_hellinger"
  python3 trainVaryLoss.py --batchsize 100 --num_units ${hu} --num_epochs 50 --num_training_files 5 --npdf 5 --lr 1e-3 --num_training_params 256 --num_layers 2 --scale_or_hyp "scale" --max_mv "update" --loss_function "joint_hellinger"
  python3 trainVaryLoss.py --batchsize 100 --num_units ${hu} --num_epochs 50 --num_training_files 5 --npdf 10 --lr 1e-3 --num_training_params 256 --num_layers 2 --scale_or_hyp "scale" --max_mv "update" --loss_function "joint_hellinger"
  

done







