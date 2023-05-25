// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __PUJ_ML__Optimizers__Base__h__
#define __PUJ_ML__Optimizers__Base__h__

#include <functional>
#include <Eigen/Core>

// -------------------------------------------------------------------------
#define PUJ_ML_Attribute( _t, _a, _i, _d )              \
  protected:                                            \
  _t m_##_a { _d };                                     \
  public:                                               \
  const _t& _i( ) const                                 \
  {                                                     \
    return( this->m_##_a );                             \
  }                                                     \
  void set_##_i( const _t& v )                          \
  {                                                     \
    this->m_##_a = v;                                   \
  }

namespace PUJ_ML
{
  namespace Optimizers
  {
    /**
     */
    template< class _C, class _X, class _Y >
    class Base
    {
    public:
      using TCost = _C;
      using TX = Eigen::EigenBase< _X >;
      using TY = Eigen::EigenBase< _Y >;

      using TModel = typename TCost::TModel;
      using TReal = typename TModel::TReal;
      using TMatrix = typename TModel::TMatrix;
      using TCol = typename TModel::TCol;
      using TRow = typename TModel::TRow;
      using MMatrix = typename TModel::MMatrix;
      using MCol = typename TModel::MCol;
      using MRow = typename TModel::MRow;
      using ConstMMatrix = typename TModel::ConstMMatrix;
      using ConstMCol = typename TModel::ConstMCol;
      using ConstMRow = typename TModel::ConstMRow;

      // Cost, gradient magnitude, batch, epoch -> should stop?
      using TSgn = bool( const TReal&, const TReal&, const unsigned long long& );
      using TDebug = std::function< TSgn >;

    public:
      PUJ_ML_Attribute( TReal, Alpha, learning_rate, 1e-2 );
      PUJ_ML_Attribute( TReal, Lambda, regularization_coefficient, 0 );
      PUJ_ML_Attribute( TReal, Epsilon, gradient_epsilon, 0 );

      PUJ_ML_Attribute( unsigned long long, BatchSize, batch_size, 0 );
      PUJ_ML_Attribute(
        unsigned long long, MaxEpochs, maximum_epochs,
        std::numeric_limits< unsigned long long >::max( )
        );
      PUJ_ML_Attribute(
        unsigned long long, DebugEpochs, debugging_epochs,
        std::numeric_limits< unsigned long long >::max( )
        );

    public:
      Base( TModel& m, const TX& X, const TY& Y );
      virtual ~Base( ) = default;

      void set_debug( TDebug d );

      void set_regularization_type_to_ridge( );
      void set_regularization_type_to_LASSO( );

      virtual void fit( ) = 0;

    protected:
      void _batches(
        std::vector< std::vector< unsigned long long > >& indices
        ) const;

    protected:
      TModel* m_Model;
      const TX* m_X;
      const TY* m_Y;

      TDebug m_Debug;
    };
  } // end namespace
} // end namespace

#include <PUJ_ML/Optimizers/Base.hxx>

#endif // __PUJ_ML__Optimizers__Base__h__

// eof - $RCSfile$
