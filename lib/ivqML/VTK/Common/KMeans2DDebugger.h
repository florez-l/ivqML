// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__VTK___Common__KMeans2DDebugger__h__
#define __ivqML__VTK___Common__KMeans2DDebugger__h__

#include <vtkChartXY.h>
#include <vtkContextView.h>
#include <vtkNamedColors.h>
#include <vtkPlotPoints.h>

#include <ivqML/VTK/Common/Table.h>


namespace ivqML
{
  namespace VTK
  {
    namespace Common
    {
      /**
       */
      template< class _TX, class _TM >
      struct KMeans2DDebugger
      {
        _TX* _X { nullptr };
        _TM* _M { nullptr };
        ivqML::VTK::Common::Table< _TX > _Xtable;
        ivqML::VTK::Common::Table< _TM > _Mtable;

        vtkSmartPointer< vtkNamedColors > _Colors;
        vtkSmartPointer< vtkContextView > _View;
        vtkSmartPointer< vtkChartXY >     _Chart;
        vtkPlot* _Xpoints { nullptr };
        vtkPlot* _Mpoints { nullptr };

        void Init( Eigen::EigenBase< _TX >& X, ::Eigen::EigenBase< _TM >& M );
        bool operator()( const float& err, const unsigned long long& iter );
        void Start( );
      };
    } // end namespace
  } // end namespace
} // end namespace

#include <ivqML/VTK/Common/KMeans2DDebugger.hxx>

#endif // __ivqML__VTK___Common__KMeans2DDebugger__h__

// eof - $RCSfile$
