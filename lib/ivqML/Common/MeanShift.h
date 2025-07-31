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
    template< class _TData, class _TReal = long double >
    class MeanShift
    {
    public:
      using TData     = _TData;
      using TDataBase = Eigen::EigenBase< TData >;

      using TReal   = _TReal;
      using TMatrix = Eigen::Matrix< TReal, Eigen::Dynamic, Eigen::Dynamic >;
      using TRow    = Eigen::Matrix< TReal, 1, Eigen::Dynamic >;

    public:
      MeanShift( const TData& D )
        : m_Data( &D )
        {
        }

      /**
       */
      void Compute( )
        {
          const auto& D = this->m_Data->derived( );
          if( this->m_IsHistogram )
          {
            auto X = D.block( 0, 0, D.rows( ) - 1, D.cols( ) );
            auto F = D.block( D.rows( ) - 1, 0, 1, D.cols( ) );

            std::cout
              << "_Z" << typeid( _TData ).name( ) << std::endl
              << "_Z" << typeid( TData ).name( ) << std::endl
              << "_Z" << typeid( X ).name( ) << std::endl
              << "_Z" << typeid( F ).name( ) << std::endl;
            this->_Compute( X, &F );
          } // end if
        }

    protected:
      void _Compute(
        const Eigen::Block< const TData >& X,
        const Eigen::Block< const TData >* F
        )
        {
          std::cout << "-------------------------------" << std::endl;
          std::cout << X << std::endl;
          std::cout << "-------------------------------" << std::endl;
          std::cout << *F << std::endl;
          std::cout << "-------------------------------" << std::endl;
        }

      /* TODO
         template< class _TData >
         static auto Histogram( const Eigen::EigenBase< _TData >& bD )
         {
         static const TReal eps = std::pow( TReal( 10 ), std::log10( std::numeric_limits< TReal >::epsilon( ) ) * 0.5 );
         unsigned long long max_iter = 100;

         auto D = bD.derived( ).template cast< TReal >( );
         auto X = D.block( 0, 0, D.rows( ), D.cols( ) - 1 );
         auto F = D.block( 0, D.cols( ) - 1, D.rows( ), 1 ).array( );
         TMatrix M = X;

         for( Eigen::Index r = 0; r < D.rows( ); ++r )
         {
         std::cout << ( r + 1 ) << "/" << D.rows( ) << std::endl;
         bool stop = false;
         unsigned long long i = 0;
         while( !stop )
         {
         i += 1;

         auto R = M.row( r );

         auto K = ( ( X.rowwise( ) - R ).array( ).pow( 2 ).rowwise( ).sum( ).array( ) / TReal( -1.5 ) ).exp( ).array( ) * F;
         TRow m = ( X.array( ).colwise( ) * K.array( ) ).colwise( ).sum( ) / K.sum( );

         TReal e = std::sqrt( ( M.row( r ) - m ).array( ).pow( 2 ).sum( ) );
         M.row( r ) = m;
         stop = ( e <= eps ) || !( i < max_iter );
         } // end if
         } // end for
         return( M );
         }
      */
      /* TODO
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
         MeanShift(
         TReal* data, const TNatural& dims, const TNatural& samples,
         TReal* frequencies = nullptr
         );
         virtual ~MeanShift( );

         template< class _TOutIt >
         void GetMeans( _TOutIt out );

         protected:
         void _init( );
         void _allocate( const TNatural& S );
         void _free( );
         void _go(
         TReal* I, TReal* O, const TNatural& N, const TNatural& M,
         TReal* F
         );

         protected:
         TReal* m_ShiftedData { nullptr };

         TMeansMap        m_MeansMap;
         TShiftedMeansMap m_ShiftedMeansMap;

         TNatural m_MaximumNumberOfIterations { 100 };
         TReal    m_DistanceError;
         TKernel  m_Kernel;
         };
      */

    protected:
      const TDataBase* m_Data { nullptr };

      bool m_IsHistogram { true };
    };
  } // end namespace
} // end namespace

// -------------------------------------------------------------------------
/* TODO
   template< class _TReal, class _TNatural >
   template< class _TOutIt >
   void ivqML::Common::MeanShift< _TReal, _TNatural >::
   GetMeans( _TOutIt out )
   {
   for( const auto& m: this->m_ShiftedMeansMap )
   for( const auto& v: m.first )
   *out = v;
   }
*/

#endif // __ivqML__Common__MeanShift__h__

// eof - $RCSfile$
