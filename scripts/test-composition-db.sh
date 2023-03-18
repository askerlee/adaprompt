#!/usr/bin/fish
#set fish_trace 1
set GPU 0
set n_iter 1
set config v1-inference.yaml
set scale 10
set outdir samples-dreambooth
fish scripts/composition-cases.sh

for case in $cases
    echo $case
    set -l case2 (string split " | " $case)
    set subject $case2[1]
    set prompt0 $case2[2]
    set folder  $case2[3]
    set class   $case2[4]
    set placeholder  "z $class"

    if test -z (string match --entire "{}" $prompt0)
        set prompt "a $placeholder $prompt0"
    else
        set prompt (string replace "{}" $placeholder $prompt0)
    end

    set ckptname  (ls -1 -rt logs|grep $subject-dreambooth|tail -1)
    if test -z "$ckptname"
    	echo Unable to find the checkpoint of $subject
    	continue
    end

    echo $subject: $ckptname $prompt
    python3 scripts/stable_txt2img.py --config configs/stable-diffusion/$config --ddim_eta 0.0 --n_samples 8 --ddim_steps 100 --ckpt logs/$ckptname/checkpoints/last.ckpt --prompt "$prompt" --gpu $GPU --scale $scale --n_iter $n_iter --outdir $outdir --indiv_subdir $folder
end
