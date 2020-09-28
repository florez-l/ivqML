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
    using TMatrix = Eigen::Matrix< _TScalar, Eigen::Dynamic, Eigen::Dynamic >;

    TMatrix operator()( const TMatrix& z, bool derivative ) const;
  };

  /**
   */
  template< class _TScalar >
  struct BinaryStep
  {
    using TMatrix = Eigen::Matrix< _TScalar, Eigen::Dynamic, Eigen::Dynamic >;

    TMatrix operator()( const TMatrix& z, bool derivative ) const;
  };

  /**
   */
  template< class _TScalar >
  struct Logistic
  {
    using TMatrix = Eigen::Matrix< _TScalar, Eigen::Dynamic, Eigen::Dynamic >;

    TMatrix operator()( const TMatrix& z, bool derivative ) const;
  };

  /**
   */
  template< class _TScalar >
  struct Tanh
  {
    using TMatrix = Eigen::Matrix< _TScalar, Eigen::Dynamic, Eigen::Dynamic >;

    TMatrix operator()( const TMatrix& z, bool derivative ) const;
  };

  /**
   */
  template< class _TScalar >
  struct ArcTan
  {
    using TMatrix = Eigen::Matrix< _TScalar, Eigen::Dynamic, Eigen::Dynamic >;

    TMatrix operator()( const TMatrix& z, bool derivative ) const;
  };

  /**
   */
  template< class _TScalar >
  struct ReLU
  {
    using TMatrix = Eigen::Matrix< _TScalar, Eigen::Dynamic, Eigen::Dynamic >;

    TMatrix operator()( const TMatrix& z, bool derivative ) const;
  };

  /**
   */
  template< class _TScalar >
  struct LeakyReLU
  {
    using TMatrix = Eigen::Matrix< _TScalar, Eigen::Dynamic, Eigen::Dynamic >;

    TMatrix operator()( const TMatrix& z, bool derivative ) const;
  };

  /**
   */
  template< class _TScalar >
  struct RandomizedReLU
  {
    using TMatrix = Eigen::Matrix< _TScalar, Eigen::Dynamic, Eigen::Dynamic >;

    TMatrix operator()( const TMatrix& z, bool derivative ) const;

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
    using TMatrix = Eigen::Matrix< _TScalar, Eigen::Dynamic, Eigen::Dynamic >;

    TMatrix operator()( const TMatrix& z, bool derivative ) const;

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
    using TMatrix = Eigen::Matrix< _TScalar, Eigen::Dynamic, Eigen::Dynamic >;

    TMatrix operator()( const TMatrix& z, bool derivative ) const;
  };

};

#endif // __PUJ_ML__ActivationFunctions__h__

// eof - $RCSfile$
