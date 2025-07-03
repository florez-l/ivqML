// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Common__MeanShift__h__
#define __ivqML__Common__MeanShift__h__

#include <ivqML/Config.h>

#include <map>
#include <vector>

namespace ivqML
{
  namespace Common
  {
    /**
     */
    template< class _TReal, class _TNatural = unsigned long long >
    class MeanShift
    {
    public:
      using Self       = MeanShift;
      using TReal      = _TReal;
      using TNatural   = _TNatural;
      using TColumn    = Eigen::Matrix< TReal, Eigen::Dynamic, 1 >;
      using TColumnMap = Eigen::Map< TColumn >;

      using TKernel = std::function< TReal( const TColumnMap&, const TColumnMap& ) >;

    protected:
      struct SShiftCmp
      {
        bool operator()( const TColumnMap& a, const TColumnMap& b ) const;
      };
      using TMeansMap = std::map< TColumnMap, TColumnMap, SShiftCmp >;
      using TShiftedMeansMap = std::map< TColumnMap, std::vector< TColumnMap >, SShiftCmp >;

    public:
      MeanShift( TReal* data, const TNatural& dims, const TNatural& samples );
      virtual ~MeanShift( );

    protected:
      void _init( );
      void _allocate( const TNatural& S );
      void _free( );
      void _go( TReal* I, TReal* O, const TNatural& N, const TNatural& M );

    protected:
      TReal* m_ShiftedData { nullptr };

      TMeansMap        m_MeansMap;
      TShiftedMeansMap m_ShiftedMeansMap;

      TNatural m_MaximumNumberOfIterations { 100 };
      TReal    m_DistanceError;
      TKernel  m_Kernel;
    };
  } // end namespace
} // end namespace

#endif // __ivqML__Common__MeanShift__h__

// eof - $RCSfile$
