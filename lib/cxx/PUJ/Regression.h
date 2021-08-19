// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ__Regression__h__
#define __PUJ__Regression__h__

#include <Eigen/Core>

namespace PUJ
{
  namespace Regression
  {
    /**
     */
    template< class _TScalar >
    class Base
    {
    public:
      using TScalar = _TScalar;
      using TRowVector = Eigen::Matrix< TScalar, 1, Eigen::Dynamic >;
      using TColumnVector = Eigen::Matrix< TScalar, Eigen::Dynamic, 1 >;
      using TMatrix = Eigen::Matrix< TScalar, Eigen::Dynamic, Eigen::Dynamic >;

    public:
      Base( const TMatrix& X, const TMatrix& y );
      virtual ~Base( ) = default;

      unsigned long NumberOfExamples( ) const;
      unsigned long VectorSize( ) const;

      virtual TScalar CostAndGradient( TRowVector& gt, const TRowVector& theta ) = 0;

    protected:
      TMatrix m_X;
      TMatrix m_y;
    };

    /**
     */
    template< class _TScalar >
    class MSE
      : public Base< _TScalar >
    {
    public:
      using Superclass = Base< _TScalar >;
      using TScalar = typename Superclass::TScalar;
      using TRowVector = typename Superclass::TRowVector;
      using TColumnVector = typename Superclass::TColumnVector;
      using TMatrix = typename Superclass::TMatrix;

    public:
      MSE( const TMatrix& X, const TMatrix& y );
      virtual ~MSE( ) = default;

      void AnalyticSolve( TRowVector& theta );
      virtual TScalar CostAndGradient( TRowVector& gt, const TRowVector& theta ) override;

    protected:
      TMatrix m_XtX;
      TMatrix m_Xby;
      TColumnVector m_uX;
      TScalar m_uy;
      TScalar m_yty;
    };

    /**
     */
    template< class _TScalar >
    class MaximumLikelihood
      : public Base< _TScalar >
    {
    public:
      using Superclass = Base< _TScalar >;
      using TScalar = typename Superclass::TScalar;
      using TRowVector = typename Superclass::TRowVector;
      using TColumnVector = typename Superclass::TColumnVector;
      using TMatrix = typename Superclass::TMatrix;

    public:
      MaximumLikelihood( const TMatrix& X, const TMatrix& y );
      virtual ~MaximumLikelihood( ) = default;

      virtual TScalar CostAndGradient( TRowVector& gt, const TRowVector& theta ) override;

    protected:
      TMatrix m_Xby;
      TScalar m_uy;
      TScalar m_Eps;
    };
  } // end namespace
} // end namespace

#endif // __PUJ__Regression__h__

// eof - $RCSfile$
