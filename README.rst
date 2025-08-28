========
tfscreen
========

Library for simulating and analyzing high-throughput screens of transcription
factor function.

Nomenclature
------------
+ *condition*: A growth condition defined by marker, selection, and iptg. A 
  genotype will have the same average growth rate in the same condition. 
+ *sample*: A tube growing under a specific growth condition. It is defined by 
  replicate, marker, selection, and iptg. 
+ *timepoint*: An aliquot of a given sample taken at a specific time. It is
  defined by replicate, marker, selection, iptg, and time.