while getopts d:p: OPT
do
  case $OPT in
    "d" ) FLG_D="TRUE" ; VALUE_D="$OPTARG" ;;
    "p" ) FLG_P="TRUE" ; VALUE_P="$OPTARG" ;;
  esac
done

g++ -o pipeline.out pipeline.cpp
for i in $(seq 1 5)
do 
    echo "random seed is $i"
    python3 ./data/prep.py -d ${VALUE_D} -p "./data/${VALUE_D}/" -s $i
    ./pipeline.out ${VALUE_P} < "data/${VALUE_D}/${VALUE_D}.in"
done