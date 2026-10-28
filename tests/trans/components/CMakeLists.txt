# (C) Copyright 2026- NVIDIA.
# Licensed under the Apache License, Version 2.0.

if( HAVE_CPU )
  add_subdirectory( cpu )
endif()

if( HAVE_GPU AND HAVE_ACC )
  add_subdirectory( gpu )
endif()
