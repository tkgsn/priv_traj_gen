#!/bin/bash

variable=$1
# 定義済みのリージョンリスト
regions=($(az account list-locations --query "[].name" -o tsv))

# リージョンごとにコンテナインスタンスをデプロイする試行
for region in "${regions[@]}"; do
  echo "Attempting to deploy in $region..."

  source $variable

  az container create \
    --resource-group $resource_group \
    --name $container_name \
    --image $image \
    --cpu $cpu --memory $memory \
    --registry-login-server $docker_repo_server \
    --registry-username $registry_user_name \
    --registry-password $registry_pass \
    --location $region \
    --restart-policy Never \
    --gitrepo-url $gitrepourl \
    --gitrepo-mount-path /$dirname \
    --environment-variables "${environment_variables[@]}" \
    --command-line "$CMD"
    # --command-line "tail -f /dev/null" # debug 

  # デプロイコマンドの結果をチェック
  if [ $? -eq 0 ]; then
    echo "Successfully deployed in $region."
    break
  else
    echo "Failed to deploy in $region. Trying next region..."
  fi
done

echo "Script finished."
