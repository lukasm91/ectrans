# (C) Copyright 2026- NVIDIA.
# Licensed under the Apache License, Version 2.0.

# Exercise the existing GPU stage interfaces without adding a public API.
if( NOT HAVE_GPU OR NOT HAVE_ACC )
  return()
endif()

foreach( precision ${precisions} )
  set( generated_dir ${CMAKE_CURRENT_BINARY_DIR}/${precision} )
  file( MAKE_DIRECTORY ${generated_dir} )
  # The global generate_file function is also defined by etrans. Select the
  # trans rules explicitly so enabling etrans cannot change these tests.
  set( rename_rules ${PROJECT_SOURCE_DIR}/src/trans/sedrenames.txt )
  file( READ ${rename_rules} rules )
  string( REPLACE VARIANTDESIGNATOR gpu_${precision} rules "${rules}" )
  if( precision STREQUAL dp )
    set( real_kind JPRD )
  else()
    set( real_kind JPRM )
  endif()
  string( TOLOWER ${real_kind} lower_kind )
  string( REPLACE TYPEDESIGNATOR_UPPER ${real_kind} rules "${rules}" )
  string( REPLACE TYPEDESIGNATOR_LOWER ${lower_kind} rules "${rules}" )
  file( WRITE ${generated_dir}/renames.sed "${rules}" )
  foreach( source component_test_utils test_grid_transpose test_fourier_transpose test_fft test_legendre )
    add_custom_command( OUTPUT ${generated_dir}/${source}.F90
      COMMAND sed -rf ${generated_dir}/renames.sed ${CMAKE_CURRENT_SOURCE_DIR}/${source}.F90
        > ${generated_dir}/${source}.F90
      DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${source}.F90 ${rename_rules} ${generated_dir}/renames.sed
      VERBATIM )
  endforeach()
  ecbuild_add_library( TARGET component_test_utils_${precision} TYPE STATIC
    SOURCES ${generated_dir}/component_test_utils.F90
    PUBLIC_LIBS trans_gpu_${precision} OpenACC::OpenACC_Fortran )
  ecbuild_target_fortran_module_directory( TARGET component_test_utils_${precision}
    MODULE_DIRECTORY ${generated_dir}/module )

  foreach( component grid_transpose fourier_transpose fft legendre )
    set( target ectrans_test_component_${component}_gpu_${precision} )
    ecbuild_add_executable( TARGET ${target}
      SOURCES ${generated_dir}/test_${component}.F90
      LIBS component_test_utils_${precision} )
    foreach( mpi ${ntasks} )
      foreach( fields 1 7 )
        set( test ${target}_mpi${mpi}_fields${fields} )
        ecbuild_add_test( TARGET ${test} COMMAND ${target} ARGS ${fields}
          MPI ${mpi} OMP 1 LABELS "gpu;components" )
        set_tests_properties( ${test} PROPERTIES RESOURCE_LOCK ectrans_gpu TIMEOUT 180 )
      endforeach()
    endforeach()
  endforeach()
endforeach()
