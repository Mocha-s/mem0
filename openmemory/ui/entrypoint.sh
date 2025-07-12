#!/bin/sh
set -e

# Ensure the working directory is correct
cd /app



# Replace env variable placeholders with real values
printenv | grep NEXT_PUBLIC_ | while IFS='=' read -r key value ; do
  # 使用 perl 替换，更安全地处理特殊字符
  find .next/ -type f -exec perl -pi -e "s/\Q$key\E/\Q$value\E/g" {} + 2>/dev/null || true
done
echo "Done replacing env variables NEXT_PUBLIC_ with real values"


# Execute the container's main process (CMD in Dockerfile)
exec "$@"