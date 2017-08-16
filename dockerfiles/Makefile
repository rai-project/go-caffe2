
default: buildlocalamd64

# Build Docker image
build: docker_build output

# Build and push Docker image
release: docker_build docker_push output

# Image and binary can be overidden with env vars.
DOCKER_IMAGE ?= carml/go-caffe2


# Get System architecture
# for now... ask carl how best to get this
ARCH ?= $(shell go env GOARCH)

# Get the latest commit.
GIT_COMMIT = $(strip $(shell git rev-parse --short HEAD))

# Get the version number from the code
CODE_VERSION = $(strip $(shell cat ../VERSION))


# Find out if the working directory is clean
GIT_NOT_CLEAN_CHECK = $(shell git status --porcelain)
ifneq (x$(GIT_NOT_CLEAN_CHECK), x)
DOCKER_TAG_SUFFIX = "-dirty"
endif



# If we're releasing to Docker Hub, and we're going to mark it with the latest tag, it should exactly match a version release
ifeq ($(MAKECMDGOALS),release)
# Use the version number as the release tag.
DOCKER_TAG = $(CODE_VERSION)

ifndef CODE_VERSION
$(error You need to create a VERSION file to build a release)
endif

# See what commit is tagged to match the version
VERSION_COMMIT = $(strip $(shell git rev-list $(CODE_VERSION) -n 1 | cut -c1-7))
ifneq ($(VERSION_COMMIT), $(GIT_COMMIT))
$(error echo You are trying to push a build based on commit $(GIT_COMMIT) but the tagged release version is $(VERSION_COMMIT))
endif

# Don't push to Docker Hub if this isn't a clean repo
ifneq (x$(GIT_NOT_CLEAN_CHECK), x)
$(error echo You are trying to release a build based on a dirty repo)
endif

else
# Add the commit ref for development builds. Mark as dirty if the working directory isn't clean
DOCKER_TAG = $(ARCH)-$(CODE_VERSION)-$(GIT_COMMIT)$(DOCKER_TAG_SUFFIX)
endif

buildlocalppc:
	# Build Docker image
	@docker build \
		--build-arg BUILD_DATE=`date -u +"%Y-%m-%dT%H:%M:%SZ"` \
		--build-arg VERSION=$(CODE_VERSION) \
		--build-arg VCS_URL=`git config --get remote.origin.url` \
		--build-arg VCS_REF=$(GIT_COMMIT) \
		-f Dockerfile.ppc64le_gpu \
		-t $(DOCKER_IMAGE):gpu-$(DOCKER_TAG) .
buildlocalamd64gpu:
	# Build Docker image
	@docker build \
		--build-arg BUILD_DATE=`date -u +"%Y-%m-%dT%H:%M:%SZ"` \
		--build-arg VERSION=$(CODE_VERSION) \
		--build-arg VCS_URL=`git config --get remote.origin.url` \
		--build-arg VCS_REF=$(GIT_COMMIT) \
		-f Dockerfile.amd64_gpu \
		-t $(DOCKER_IMAGE):gpu-$(DOCKER_TAG) .
buildlocalamd64cpu:
	# Build Docker image
	@docker build \
		--build-arg BUILD_DATE=`date -u +"%Y-%m-%dT%H:%M:%SZ"` \
		--build-arg VERSION=$(CODE_VERSION) \
		--build-arg VCS_URL=`git config --get remote.origin.url` \
		--build-arg VCS_REF=$(GIT_COMMIT) \
		-f Dockerfile.amd64_cpu \
		-t "$(DOCKER_IMAGE):cpu-$(DOCKER_TAG)" .

buildlocalamd64: buildlocalamd64cpu


docker_push:
	# Tag image as latest
	docker tag $(DOCKER_IMAGE):cpu-$(DOCKER_TAG) $(DOCKER_IMAGE):cpu-latest
	docker tag $(DOCKER_IMAGE):gpu-$(DOCKER_TAG) $(DOCKER_IMAGE):gpu-latest

	# Push to DockerHub
	docker push $(DOCKER_IMAGE):cpu-$(DOCKER_TAG)
	docker push $(DOCKER_IMAGE):gpu-$(DOCKER_TAG)
	docker push $(DOCKER_IMAGE):cpu-latest
	docker push $(DOCKER_IMAGE):gpu-latest

output:
	@echo Docker Image: $(DOCKER_IMAGE):$(DOCKER_TAG)


## See https://github.com/microscaling/microscaling/blob/master/Makefile
