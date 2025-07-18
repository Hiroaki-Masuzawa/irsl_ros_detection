
abs_script=$(readlink -f "$0")
abs_dir=$(dirname ${abs_script})

docker pull repo.irsl.eiiris.tut.ac.jp/irsl_base:cuda_12.1.0-cudnn8-devel-ubuntu22.04_one
docker build . -f ${abs_dir}/Dockerfile -t irsl_object_perception
