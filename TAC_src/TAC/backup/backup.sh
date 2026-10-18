
echo -n "input backup_name: "

read backup_name

mkdir ${backup_name}

cd ${backup_name}

rsync -av --exclude="*.ncu-rep" --exclude="*.nsys-rep" --exclude="build" ../../sampling_server .
rsync -av --exclude="*.ncu-rep" --exclude="*.nsys-rep" --exclude="build" ../../training_backend .