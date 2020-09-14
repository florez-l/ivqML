// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__ActivationFunctions__h__
#define __PUJ_ML__ActivationFunctions__h__

#include <Eigen/Core>

namespace ActivationFunctions
{
  /**
   */
  template< class _TScalar >
  struct Identity
  {
    using TColVector = Eigen::Matrix< _TScalar, Eigen::Dynamic, 1 >;

    TColVector operator()( const TColVector& z ) const;
    TColVector operator[]( const TColVector& z ) const;
  };

  /**
   */
  template< class _TScalar >
  struct BinaryStep
  {
    using TColVector = Eigen::Matrix< _TScalar, Eigen::Dynamic, 1 >;

    TColVector operator()( const TColVector& z ) const;
    TColVector operator[]( const TColVector& z ) const;
  };

  /**
   */
  template< class _TScalar >
  struct Logistic
  {
    using TColVector = Eigen::Matrix< _TScalar, Eigen::Dynamic, 1 >;

    TColVector operator()( const TColVector& z ) const;
    TColVector operator[]( const TColVector& z ) const;
  };

  /**
   */
  template< class _TScalar >
  struct Tanh
  {
    using TColVector = Eigen::Matrix< _TScalar, Eigen::Dynamic, 1 >;

    TColVector operator()( const TColVector& z ) const;
    TColVector operator[]( const TColVector& z ) const;
  };

  /**
   */
  template< class _TScalar >
  struct ArcTan
  {
    using TColVector = Eigen::Matrix< _TScalar, Eigen::Dynamic, 1 >;

    TColVector operator()( const TColVector& z ) const;
    TColVector operator[]( const TColVector& z ) const;
  };

  /**
   */
  template< class _TScalar >
  struct ReLU
  {
    using TColVector = Eigen::Matrix< _TScalar, Eigen::Dynamic, 1 >;

    TColVector operator()( const TColVector& z ) const;
    TColVector operator[]( const TColVector& z ) const;
  };

  /**
   */
  template< class _TScalar >
  struct LeakyReLU
  {
    using TColVector = Eigen::Matrix< _TScalar, Eigen::Dynamic, 1 >;

    TColVector operator()( const TColVector& z ) const;
    TColVector operator[]( const TColVector& z ) const;
  };

  /**
   */
  template< class _TScalar >
  struct RandomizedReLU
  {
    using TColVector = Eigen::Matrix< _TScalar, Eigen::Dynamic, 1 >;

    TColVector operator()( const TColVector& z ) const;
    TColVector operator[]( const TColVector& z ) const;

    RandomizedReLU( const _TScalar& a );
    const _TScalar& GetA( ) const;

  protected:
    _TScalar m_A;
  };

  /**
   */
  template< class _TScalar >
  struct ELU
  {
    using TColVector = Eigen::Matrix< _TScalar, Eigen::Dynamic, 1 >;

    TColVector operator()( const TColVector& z ) const;
    TColVector operator[]( const TColVector& z ) const;

    ELU( const _TScalar& a );
    const _TScalar& GetA( ) const;

  protected:
    _TScalar m_A;
  };

  /**
   */
  template< class _TScalar >
  struct SoftPlus
  {
    using TColVector = Eigen::Matrix< _TScalar, Eigen::Dynamic, 1 >;

    TColVector operator()( const TColVector& z ) const;
    TColVector operator[]( const TColVector& z ) const;
  };

};

#endif // __PUJ_ML__ActivationFunctions__h__

// eof - $RCSfile$
