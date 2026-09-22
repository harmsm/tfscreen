rm -rf ~/miniconda3/lib/python3.13/site-packages/tfscreen* 
rm -rf ~/miniconda3/bin/tfs-*
rm -rf build/ 
rm -f ~/miniconda3/lib/python3.13/site-packages/__editable__.tfscreen-*.pth
for x in `find . -iname "__pycache__"`; do rm -rf $x; done
