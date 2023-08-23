FROM yic/rad_nerf_training:base

# Clone common repo to get other integration utils
RUN mkdir -m 700 /root/.ssh; \
  touch -m 600 /root/.ssh/known_hosts; \
  ssh-keyscan github.com > /root/.ssh/known_hosts

WORKDIR /workspace/RAD-NeRF/
RUN --mount=type=ssh,id=github git clone -b master git@github.com:yic03685/LearningBuddyCommon.git

WORKDIR /workspace/RAD-NeRF/

