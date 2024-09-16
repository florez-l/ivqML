// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__VTK___Common__Table__h__
#define __ivqML__VTK___Common__Table__h__

#include <vtkFloatArray.h>
#include <vtkSmartPointer.h>
#include <vtkTable.h>

namespace ivqML
{
  namespace VTK
  {
    namespace Common
    {
      /**
       */
      template< class _TM >
      struct Table
      {
        float* _Buffer { nullptr };
        std::vector< vtkSmartPointer< vtkFloatArray > > _Arrays;
        vtkSmartPointer< vtkTable > _Table;

        void Init( _TM& D );
        void Modified( );
      };
    } // end namespace
  } // end namespace
} // end namespace

#include <ivqML/VTK/Common/Table.hxx>

#endif // __ivqML__VTK___Common__Table__h__

// eof - $RCSfile$
