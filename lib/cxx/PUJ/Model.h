// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ__Model__h__
#define __PUJ__Model__h__

#include <Eigen/Core>

namespace PUJ
{
  namespace Model
  {
    /**
     */
    template< class _TScalar >
    class Linear
    {
    public:
      using TScalar = _TScalar;
      using TRowVector = Eigen::Matrix< TScalar, 1, Eigen::Dynamic >;
      using TColumnVector = Eigen::Matrix< TScalar, Eigen::Dynamic, 1 >;
      using TMatrix = Eigen::Matrix< TScalar, Eigen::Dynamic, Eigen::Dynamic >;

    public:
      Linear( const TRowVector& w, const TScalar& b );
      virtual ~Linear( ) = default;

      unsigned long Dimensions( ) const;
      const TRowVector& Weights( ) const;
      const TScalar& Bias( ) const;

      void SetWeights( const TRowVector& w );
      void SetBias( const TScalar& b );

      TMatrix operator()( const TMatrix& x ) const;

    protected:
      TRowVector m_Weights;
      TScalar    m_Bias;
    };

    /**
     */
    template< class _TScalar >
    class Logistic
      : public Linear< _TScalar >
    {
    public:
      using Superclass = Linear< _TScalar >;
      using TScalar = typename Superclass::TScalar;
      using TRowVector = typename Superclass::TRowVector;
      using TColumnVector = typename Superclass::TColumnVector;
      using TMatrix = typename Superclass::TMatrix;

    public:
      Logistic( const TRowVector& w, const TScalar& b );
      virtual ~Logistic( ) = default;

      TMatrix operator()( const TMatrix& x, bool threshold = true ) const;
    };
  } // end namespace
} // end namespace

#endif // __PUJ__Model__h__

// eof - $RCSfile$
