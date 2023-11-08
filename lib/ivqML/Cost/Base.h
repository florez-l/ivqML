// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Cost__Base__h__
#define __ivqML__Cost__Base__h__

#include <utility>

// -------------------------------------------------------------------------
#define ivqML_Cost_Typedefs                             \
  public:                                               \
  using TModel = typename Superclass::TModel;           \
  using TDX = typename Superclass::TDX;                 \
  using TDY = typename Superclass::TDY;                 \
  using TX = typename Superclass::TX;                   \
  using TY = typename Superclass::TY;                   \
  using TScalar = typename Superclass::TScalar;         \
  using TNatural = typename Superclass::TNatural;       \
  using TMatrix = typename Superclass::TMatrix;         \
  using TMap = typename Superclass::TMap;               \
  using TConstMap = typename Superclass::TConstMap;     \
  using TResult = typename Superclass::TResult
  
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
      virtual ~Base( ) = default;

      TNatural number_of_batches( ) const;
      void set_batch_size( const TNatural& bs );
      virtual TResult operator()( const TNatural& batch = 0 ) = 0;

    protected:
      const _M* m_M { nullptr };
      const TX* m_X { nullptr };
      const TY* m_Y { nullptr };

      std::vector< std::pair< TNatural, TNatural > > m_B;

      TMatrix m_G;
      TMatrix m_Z;
    };
  } // end namespace
} // end namespace

#include <ivqML/Cost/Base.hxx>

#endif // __ivqML__Cost__Base__h__

// eof - $RCSfile$
