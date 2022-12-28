kill -9 `ps -ef | grep "kyowon" | awk '{print $2}'`
kill -9 `ps -ef | grep "cleaning" | grep "wandb" | awk '{print $2}'`