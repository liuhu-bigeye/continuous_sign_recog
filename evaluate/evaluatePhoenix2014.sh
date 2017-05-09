#!/bin/bash

if [ -z "$2" ];then

echo "evaluateWER.sh <hypothesis-CTM-file> <dev | test>"
exit 0
fi

hypothesisCTM=$1
partition=$2
path=/home/runpeng/workspace/C3D_CTC/evaluate
save=/home/runpeng/workspace/C3D_CTC/results
sclitepath=/home/trunk/disk1/database-rwth-2014/phoenix2014-release/evaluation/sctk-2.4.0/bin

# apply some simplifications to the recognition
cat $save/${hypothesisCTM} | sed -e 's,loc-,,g' -e 's,cl-,,g' -e 's,qu-,,g' -e 's,poss-,,g' -e 's,lh-,,g' -e 's,S0NNE,SONNE,g' -e 's,HABEN2,HABEN,g'|sed -e 's,__EMOTION__,,g' -e 's,__PU__,,g'  -e 's,__LEFTHAND__,,g' |sed -e 's,WIE AUSSEHEN,WIE-AUSSEHEN,g' -e 's,ZEIGEN ,ZEIGEN-BILDSCHIRM ,g' -e 's,ZEIGEN$,ZEIGEN-BILDSCHIRM,' -e 's,^\([A-Z]\) \([A-Z][+ ]\),\1+\2,g' -e 's,[ +]\([A-Z]\) \([A-Z]\) , \1+\2 ,g'| sed -e 's,\([ +][A-Z]\) \([A-Z][ +]\),\1+\2,g'|sed -e 's,\([ +][A-Z]\) \([A-Z][ +]\),\1+\2,g'|  sed -e 's,\([ +][A-Z]\) \([A-Z][ +]\),\1+\2,g'|sed -e 's,\([ +]SCH\) \([A-Z][ +]\),\1+\2,g'|sed -e 's,\([ +]NN\) \([A-Z][ +]\),\1+\2,g'| sed -e 's,\([ +][A-Z]\) \(NN[ +]\),\1+\2,g'| sed -e 's,\([ +][A-Z]\) \([A-Z]\)$,\1+\2,g'|  sed -e 's,\([A-Z][A-Z]\)RAUM,\1,g'| sed -e 's,-PLUSPLUS,,g' | perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| grep -v "__LEFTHAND__" | grep -v "__EPENTHESIS__" | grep -v "__EMOTION__" > $save/tmp.ctm 

#make sure empty recognition results get filled with [EMPTY] tags - so that the alignment can work out on all data.
cat $save/tmp.ctm | sed -e 's,\s*$,,'   | awk 'BEGIN{lastID="";lastRow=""}{if (lastID!=$1 && cnt[lastID]<1 && lastRow!=""){print lastRow" [EMPTY]";}if ($5!=""){cnt[$1]+=1;print $0;}lastID=$1;lastRow=$0}' > $save/tmp2.ctm
python $path/sort_file.py $save/tmp2.ctm

cat $path/phoenix2014-groundtruth-$partition.stm > $save/tmp.stm
python $path/sort_file.py $save/tmp.stm

#add missing entries, so that sclite can generate alignment
python $path/mergectmstm.py $save/tmp2.ctm $save/tmp.stm  

mv $save/tmp2.ctm $save/out.${hypothesisCTM}

#make sure NIST sclite toolbox is installed and on path. Available at ftp://jaguar.ncsl.nist.gov/pub/sctk-2.4.0-20091110-0958.tar.bz2
$sclitepath/sclite  -h $save/out.$hypothesisCTM ctm -r $save/tmp.stm stm -f 0 -o sgml sum rsum pra    
$sclitepath/sclite  -h $save/out.$hypothesisCTM ctm -r $save/tmp.stm stm -f 0 -o dtl stdout |grep Error
