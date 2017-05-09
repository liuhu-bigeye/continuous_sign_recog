import os
import sys
import shutil
import re
import commands
import tempfile

# SCLITE_PATH = '/home/trunk/disk1/database-rwth-2014/phoenix2014-release/evaluation/sctk-2.4.0/bin'
SCLITE_PATH = '/home/liuhu/tools/kaldi_sctk/bin'

def evaluate(hypotheses, vocabulary, path, epoch, phase='test'):
    path = '_'.join([path, phase])

    gt_file = '/home/liuhu/workspace/journal/evaluate/phoenix2014-groundtruth-%s.stm' % phase
    hypo_file = os.path.join(path, 'epoch_%03d.ctm' % epoch)

    # save temporary ctm file
    temp_file = tempfile.NamedTemporaryFile(delete=False, dir=path)
    temp_save = temp_file.name
    temp_file.close()

    generate_ctm_file(hypotheses, vocabulary, temp_save)
    temp_ctm = os.path.join(path, 'temp.ctm')
    temp2_ctm = os.path.join(path, 'temp2.ctm')
    temp_stm = os.path.join(path, 'temp.stm')

    cmd = '/home/liuhu/workspace/journal/evaluate/refine_ctm.sh %s %s %s' % (temp_save, temp_ctm, temp2_ctm)
    os.system(cmd)
    # cmd = """cat %s | sed -e 's,loc-,,g' -e 's,cl-,,g' -e 's,qu-,,g' -e 's,poss-,,g' -e 's,lh-,,g' -e 's,S0NNE,SONNE,g' -e 's,HABEN2,HABEN,g'|sed -e 's,__EMOTION__,,g' -e 's,__PU__,,g'  -e 's,__LEFTHAND__,,g' |sed -e 's,WIE AUSSEHEN,WIE-AUSSEHEN,g' -e 's,ZEIGEN ,ZEIGEN-BILDSCHIRM ,g' -e 's,ZEIGEN$,ZEIGEN-BILDSCHIRM,' -e 's,^\([A-Z]\) \([A-Z][+ ]\),\1+\2,g' -e 's,[ +]\([A-Z]\) \([A-Z]\) , \1+\2 ,g'| sed -e 's,\([ +][A-Z]\) \([A-Z][ +]\),\1+\2,g'|sed -e 's,\([ +][A-Z]\) \([A-Z][ +]\),\1+\2,g'|  sed -e 's,\([ +][A-Z]\) \([A-Z][ +]\),\1+\2,g'|sed -e 's,\([ +]SCH\) \([A-Z][ +]\),\1+\2,g'|sed -e 's,\([ +]NN\) \([A-Z][ +]\),\1+\2,g'| sed -e 's,\([ +][A-Z]\) \(NN[ +]\),\1+\2,g'| sed -e 's,\([ +][A-Z]\) \([A-Z]\)$,\1+\2,g'|  sed -e 's,\([A-Z][A-Z]\)RAUM,\1,g'| sed -e 's,-PLUSPLUS,,g' | perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| perl -ne 's,(?<![\w-])(\b[A-Z]+(?![\w-])) \1(?![\w-]),\1,g;print;'| grep -v "__LEFTHAND__" | grep -v "__EPENTHESIS__" | grep -v "__EMOTION__" > %s""" % (temp_save, temp_ctm)
    # os.system(cmd)

    # cmd = """cat %s | sed -e 's,\s*$,,' | awk 'BEGIN{lastID="";lastRow=""}{if (lastID!=$1 && cnt[lastID]<1 && lastRow!=""){print lastRow" [EMPTY]";}if ($5!=""){cnt[$1]+=1;print $0;}lastID=$1;lastRow=$0}' > %s""" % (temp_ctm, temp2_ctm)
    # os.system(cmd)
    sort_file(temp2_ctm)

    shutil.copyfile(gt_file, temp_stm)
    sort_file(temp_stm)

    merge_file(temp2_ctm, temp_stm)
    os.rename(temp2_ctm, hypo_file)

    cmd = '%s/sclite -h %s ctm -r %s stm -f 0 -o sum pra dtl' % (SCLITE_PATH, hypo_file, temp_stm)
    os.system(cmd)

    cmd = '%s/sclite -h %s ctm -r %s stm -f 0 -o dtl stdout' % (SCLITE_PATH, hypo_file, temp_stm)
    _, output = commands.getstatusoutput(cmd)

    obj = re.search('Percent Total Error.+\(([0-9]+)\)', output, re.M)
    num_err = int(obj.group(1))

    obj = re.search('Ref\. words.+\(([0-9]+)\)', output, re.M)
    num_total = int(obj.group(1))

    WER = float(num_err) / float(num_total)

    os.remove(temp_save)
    os.remove(temp_ctm)
    os.remove(temp_stm)

    return WER

def sort_file(loadFile):
    if os.path.basename(loadFile).split('.')[-1] == 'ctm':
        with open(loadFile, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
            lines = sorted(lines, key = lambda x: (x.split(' ')[0], float(x.split(' ')[2])))
    elif os.path.basename(loadFile).split('.')[-1] == 'stm':
        with open(loadFile, 'r') as f:
            lines = [line.strip() for line in f.readlines()]
            lines = sorted(lines, key = lambda x: x.split(' ')[0])
    else:
        raise ValueError('unsupported file type')

    with open(loadFile, 'w+') as f:
        for line in lines:
            f.write(line + '\n')


def merge_file(ctmFile, stmFile):
    with open(ctmFile, 'r') as f:
        lines = [line.strip() for line in f.readlines()]
        ctmDict_raw = [line.split()[0] for line in lines]
        ctmDict = list(set(ctmDict_raw))

    with open(stmFile, 'r') as f:
        stmDict = [line.strip().split()[0] for line in f.readlines()]

    empty_list = [item for item in stmDict if not (item in ctmDict)]


    for item in empty_list:
        k = stmDict.index(item)
        if k == len(stmDict) - 1:
            lines.append('%s 1 0.000 0.030 [EMPTY]' % item)
        else:
            j = k + 1
            item_next = stmDict[j]
            while (item_next not in ctmDict_raw) and (j < len(stmDict) - 1):
                j += 1
                item_next = stmDict[j]

            idx = ctmDict_raw.index(item_next) if item_next in ctmDict_raw else len(lines)
            lines.insert(idx, '%s 1 0.000 0.030 [EMPTY]' % item)

        ctmDict_raw = [line.split()[0] for line in lines]


    with open(ctmFile, 'w+') as f:
        for line in lines:
            f.write(line + '\n')


def convert_pred_to_hypo(prediction, h_len=None):
    hypo = []
    BLANK = 0
    for i in range(prediction.shape[0]):
        pred = prediction[i]
        h = []
        N = h_len[i] if not h_len is None else pred.shape[0]
        for j in range(N):
            if (j == 0 or pred[j-1] != pred[j]) and pred[j] != BLANK:
                h.append(pred[j])
        hypo.append(h)

    return hypo

def generate_ctm_file(hypotheses, vocabulary, ctm_file):
    with open(ctm_file, 'w') as f:
        for ID, hypo in hypotheses.items():
            for k, y in enumerate(hypo):
                splits = ID.split('/')
                filename = splits[0]
                channel = splits[1]

                f.write('%s %s %.2f %.2f %s\n'
                        % (filename, channel, .1*float(k), .1, vocabulary[int(y) - 1]))
