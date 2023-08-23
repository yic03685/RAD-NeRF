DOCKER_BUILDKIT=0
docker build \
  --no-cache \
  --progress=plain \
  -t yic/rad_nerf_training:base \
  -f docker/base/Dockerfile \
  .
