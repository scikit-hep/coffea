FROM uccross/skyhookdm-arrow:v0.4.0

RUN sed -i 's/mirrorlist/#mirrorlist/g' /etc/yum.repos.d/CentOS-Linux-* && \
    sed -i 's|#baseurl=http://mirror.centos.org|baseurl=http://vault.centos.org|g' /etc/yum.repos.d/CentOS-Linux-* && \
    yum -y update && \
    yum -y install python3 \
                   python3-devel \
                   python3-pip \
                   llvm-devel && \
    pip3 install pip --upgrade
