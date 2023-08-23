DOCKER_BUILDKIT=0
docker build --ssh \
  github=./.ssh/id_ed25519 \
  --no-cache \
  --progress=plain \
  -t yic/rad_nerf_training:main \
  -f docker/main/Dockerfile \
  .
