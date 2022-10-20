file(REMOVE_RECURSE
  "libIMSRG.dylib"
  "libIMSRG.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/IMSRG.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
