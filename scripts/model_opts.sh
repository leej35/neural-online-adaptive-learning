
if [ "${model_type}" == "GRU" ] 
then
    model_opt=" --rnn-type GRU"
elif [ "${model_type}" == "RETAIN" ] 
then
    model_opt=" --rnn-type retain"
elif [ "${model_type}" == "CNN" ] 
then
    model_opt=" --rnn-type CNN"
fi