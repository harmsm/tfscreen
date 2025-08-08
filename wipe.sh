rm -rf ~/miniconda3/lib/python3.12/site-packages/tfscreen* 
rm -rf build/ 
for x in `find . -iname "__pycache__"`; do rm -rf $x; done
