// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__ActivationFunctions__h__
#define __PUJ_ML__ActivationFunctions__h__

#include <functional>
#include <map>
#include <string>
#include <Eigen/Core>

namespace ActivationFunctions
{
  /**
   */
  template< class _TScl >
  class Function
  {
  public:
    using Self       = Function;
    using TScalar    = _TScl;
    using TColVector = Eigen::Matrix< _TScl, -1, 1 >;
    using TRowVector = Eigen::Matrix< _TScl, 1, -1 >;
    using TMatrix    = Eigen::Matrix< _TScl, -1, -1 >;

  public:
    const std::string& name( ) const;
    const TScalar& threshold( ) const;

    virtual Self* copy( ) const = 0;
    virtual TMatrix f( const TMatrix& z ) const = 0;
    virtual TMatrix d( const TMatrix& z ) const = 0;
    virtual TMatrix t( const TMatrix& z ) const;

  protected:
    std::string m_N;
    TScalar     m_T;
  };

  /**
   */
  template< class _TScl >
  class Factory
  {
  public:
    using Self       = Factory;
    using TFunction  = ActivationFunctions::Function< _TScl >;
    using TScalar    = typename TFunction::TScalar;
    using TColVector = typename TFunction::TColVector;
    using TRowVector = typename TFunction::TRowVector;
    using TMatrix    = typename TFunction::TMatrix;

    using TCreator = std::function< TFunction*( ) >;
    using TMap = std::map< std::string, TCreator >;

  public:
    Factory( );
    Factory( const Self& other );
    virtual ~Factory( );
    Self& operator=( const Self& other );

    static Self* get( );
    void reg_cre( const std::string& n, TCreator c );
    TFunction* create( const std::string& n ) const;

  protected:
    TMap m_FactoryMap;
  };
} // end namespace

// -------------------------------------------------------------------------
#define PUJ_ML_ActivationFunction( _n_, _t_ )                   \
  template< class _TScl >                                       \
  class _n_                                                     \
    : public ActivationFunctions::Function< _TScl >             \
  {                                                             \
  public:                                                       \
    using Self       = _n_;                                     \
    using Superclass = ActivationFunctions::Function< _TScl >;  \
    using TColVector = typename Superclass::TColVector;         \
    using TRowVector = typename Superclass::TRowVector;         \
    using TMatrix    = typename Superclass::TMatrix;            \
    using TScalar    = typename Superclass::TScalar;            \
    static Superclass* create( ) { return( new Self( ) ); }     \
    virtual Superclass* copy( ) const override                  \
    {                                                           \
      Self* n = new Self( );                                    \
      *n = *this;                                               \
      return( n );                                              \
    }                                                           \
    virtual TMatrix f( const TMatrix& z ) const override;       \
    virtual TMatrix d( const TMatrix& z ) const override;       \
    _n_( )                                                      \
      : Superclass( )                                           \
    {                                                           \
      this->m_N = #_n_;                                         \
      this->m_T = _t_;                                          \
    }                                                           \
  }

// -------------------------------------------------------------------------
#define PUJ_ML_ActivationFunction_1( _n_, _p_, _d_, _t_ )       \
  template< class _TScl >                                       \
  class _n_                                                     \
    : public ActivationFunctions::Function< _TScl >             \
  {                                                             \
  public:                                                       \
    using Self       = _n_;                                     \
    using Superclass = ActivationFunctions::Function< _TScl >;  \
    using TColVector = typename Superclass::TColVector;         \
    using TRowVector = typename Superclass::TRowVector;         \
    using TMatrix    = typename Superclass::TMatrix;            \
    using TScalar    = typename Superclass::TScalar;            \
    static Superclass* create( ) { return( new Self( ) ); }     \
    virtual Superclass* copy( ) const override                  \
    {                                                           \
      Self* n = new Self( );                                    \
      *n = *this;                                               \
      return( n );                                              \
    }                                                           \
    virtual TMatrix f( const TMatrix& z ) const override;       \
    virtual TMatrix d( const TMatrix& z ) const override;       \
    const _TScl& Get##_p_( ) const                              \
    {                                                           \
      return( this->m_##_p_ );                                  \
    }                                                           \
    void Set##_p_( const _TScl& v )                             \
    {                                                           \
      this->m_##_p_ = v;                                        \
    }                                                           \
    _n_( )                                                      \
      : Superclass( )                                           \
    {                                                           \
      this->m_N = #_n_;                                         \
      this->m_T = _t_;                                          \
      this->m_##_p_ = _d_;                                      \
    }                                                           \
  protected:                                                    \
    _TScl m_##_p_;                                              \
  }

namespace ActivationFunctions
{
  // -- Simple functions
  PUJ_ML_ActivationFunction( ArcTan, 0 );
  PUJ_ML_ActivationFunction( BinaryStep, 0 );
  PUJ_ML_ActivationFunction( Identity, 0 );
  PUJ_ML_ActivationFunction( LeakyReLU, 0 );
  PUJ_ML_ActivationFunction( Logistic, 0.5 );
  PUJ_ML_ActivationFunction( OutTanh, 0.5 );
  PUJ_ML_ActivationFunction( ReLU, 0 );
  PUJ_ML_ActivationFunction( SoftPlus, 0 );
  PUJ_ML_ActivationFunction( Tanh, 0 );

  // -- Functions with one parameter
  PUJ_ML_ActivationFunction_1( ELU, Alpha, 0, 0 );
  PUJ_ML_ActivationFunction_1( RandomizedReLU, Alpha, 0, 0 );
} // end namespace

#endif // __PUJ_ML__ActivationFunctions__h__

// eof - $RCSfile$
