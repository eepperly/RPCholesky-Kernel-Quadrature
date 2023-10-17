# QM9 dataset
if [ ! -d "molecules" ] 
then
    wget https://figshare.com/ndownloader/files/3195389
    mkdir -p molecules
    cd molecules
    mv ../3195389 .
    tar -xvf 3195389
    rm 3195389
    cd ..
fi

python load_qm9_data.py
