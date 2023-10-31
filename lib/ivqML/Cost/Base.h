// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Cost__Base__h__
#define __ivqML__Cost__Base__h__

/* TODO
   #include <Eigen/Core>
   #include <cmath>
   #include <ostream>
   #include <limits>
*/
#include <utility>
   
namespace ivqML
{
  namespace Cost
  {
    /**
     */
    template< class _M, class _X, class _Y >
    class Base
    {
    public:
      using Self = Base;
      using TModel = _M;
      using TDX = _X;
      using TDY = _Y;
      using TX = Eigen::EigenBase< _X >;
      using TY = Eigen::EigenBase< _Y >;
      using TScalar = typename _M::TScalar;
      using TNatural = typename _M::TNatural;
      using TMatrix = typename _M::TMatrix;
      using TMap = typename _M::TMap;
      using TConstMap = typename _M::TConstMap;

      using TResult = std::pair< TScalar, const TScalar* >;

    public:
      Base( const _M& m, const TX& iX, const TY& iY );
      virtual ~Base( );

      virtual TResult operator()( ) const = 0;


    protected:
      const _M* m_M { nullptr };
      const TX* m_X { nullptr };
      const TY* m_Y { nullptr };

      TScalar* m_G  { nullptr };
      TScalar* m_Ym { nullptr };
    };
  } // end namespace
} // end namespace

#include <ivqML/Cost/Base.hxx>

#endif // __ivqML__Cost__Base__h__

// eof - $RCSfile$