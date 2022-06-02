while getopts d: OPT
do
  case $OPT in
    "d" ) FLG_D="TRUE" ; VALUE_D="$OPTARG" ;;
  esac
done

g++ -o pipeline.out pipeline.cpp
for i in $(seq 1 2)
do 
    echo "random seed is $i"
    python3 ./data/prep.py -d ${VALUE_D} -p "./data/${VALUE_D}/" -s $i
    ./pipeline.out < "data/${VALUE_D}/${VALUE_D}.in"
done