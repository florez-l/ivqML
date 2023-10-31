// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Cost__CrossEntropy__h__
#define __ivqML__Cost__CrossEntropy__h__

#include <ivqML/Cost/Base.h>
   
namespace ivqML
{
  namespace Cost
  {
    /**
     */
    template< class _M, class _X = typename _M::TMatrix, class _Y = typename _M::TMatrix >
    class CrossEntropy
      : public ivqML::Cost::Base< _M, _X, _Y >
    {
    public:
      using Self = CrossEntropy;
      using Superclass = ivqML::Cost::Base< _M, _X, _Y >;
      using TModel = typename Superclass::TModel;
      using TDX = typename Superclass::TDX;
      using TDY = typename Superclass::TDY;
      using TX = typename Superclass::TX;
      using TY = typename Superclass::TY;
      using TScalar = typename Superclass::TScalar;
      using TNatural = typename Superclass::TNatural;
      using TMatrix = typename Superclass::TMatrix;
      using TMap = typename Superclass::TMap;
      using TConstMap = typename Superclass::TConstMap;
      using TResult = typename Superclass::TResult;

    public:
      CrossEntropy( const _M& m, const TX& iX, const TY& iY );
      virtual ~CrossEntropy( );

      virtual TResult operator()( ) const override;

    protected:
      struct
      {
        std::vector< Eigen::Index > zeros, ones;
        void init(
          const TScalar& y, const Eigen::Index& i, const Eigen::Index& j
          )
          {
            zeros.clear( );
            ones.clear( );
            this->operator()( y, i, j );
          }
        void operator()(
          const TScalar& y, const Eigen::Index& i, const Eigen::Index& j
          )
          {
            if( y == TScalar( 0 ) )
              zeros.push_back( i );
            else
              ones.push_back( i );
          }
      } m_YVisitor;
    };
  } // end namespace
} // end namespace

#include <ivqML/Cost/CrossEntropy.hxx>

#endif // __ivqML__Cost__CrossEntropy__h__

// eof - $RCSfile$
