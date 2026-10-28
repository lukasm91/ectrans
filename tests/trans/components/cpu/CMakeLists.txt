# (C) Copyright 2026- NVIDIA.
# Licensed under the Apache License, Version 2.0.

foreach( precision ${precisions} )
  set( generated_dir ${CMAKE_CURRENT_BINARY_DIR}/${precision} )
  file( MAKE_DIRECTORY ${generated_dir} )
  set( rename_rules ${PROJECT_SOURCE_DIR}/src/trans/sedrenames.txt )
  file( READ ${rename_rules} rules )
  string( REPLACE VARIANTDESIGNATOR ${precision} rules "${rules}" )
  if( precision STREQUAL dp )
    set( real_kind JPRD )
  else()
    set( real_kind JPRM )
  endif()
  string( TOLOWER ${real_kind} lower_kind )
  string( REPLACE TYPEDESIGNATOR_UPPER ${real_kind} rules "${rules}" )
  string( REPLACE TYPEDESIGNATOR_LOWER ${lower_kind} rules "${rules}" )
  file( WRITE ${generated_dir}/renames.sed "${rules}" )
  file( APPEND ${generated_dir}/renames.sed
    "s/CPU_COMPONENT_TEST_UTILS/CPU_COMPONENT_TEST_UTILS_${precision}/g\n" )
  foreach( source component_test_utils test_fft test_grid_transpose test_fourier_transpose test_legendre )
    add_custom_command( OUTPUT ${generated_dir}/${source}.F90
      COMMAND sed -rf ${generated_dir}/renames.sed ${CMAKE_CURRENT_SOURCE_DIR}/${source}.F90
        > ${generated_dir}/${source}.F90
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${source}.F90 ${rename_rules} ${generated_dir}/renames.sed
      VERBATIM )
  endforeach()
  ecbuild_add_library( TARGET cpu_component_test_utils_${precision} TYPE STATIC
    SOURCES ${generated_dir}/component_test_utils.F90 PUBLIC_LIBS trans_${precision} )
  ecbuild_target_fortran_module_directory( TARGET cpu_component_test_utils_${precision}
    MODULE_DIRECTORY ${generated_dir}/module )
  foreach( component fft grid_transpose fourier_transpose legendre )
    set( target ectrans_test_component_${component}_cpu_${precision} )
    ecbuild_add_executable( TARGET ${target} SOURCES ${generated_dir}/test_${component}.F90
      LIBS cpu_component_test_utils_${precision} )
    set( batches 0 )
    if( component STREQUAL fft )
      set( batches 0 1 )
    endif()
    foreach( mpi ${ntasks} )
      # MPL_INIT requires a launcher with a real MPI implementation.
      if( HAVE_MPI AND mpi EQUAL 0 )
        continue()
      endif()
      foreach( fields 1 7 )
        foreach( batch ${batches} )
          set( test ${target}_mpi${mpi}_fields${fields} )
          set( args ${fields} )
          if( component STREQUAL fft )
            string( APPEND test _batch${batch} )
            list( APPEND args ${batch} )
          endif()
          ecbuild_add_test( TARGET ${test} COMMAND ${target} ARGS ${args}
            MPI ${mpi} OMP 1 LABELS "cpu;components" )
          set_tests_properties( ${test} PROPERTIES TIMEOUT 180 )
        endforeach()
      endforeach()
    endforeach()
  endforeach()
endforeach()
