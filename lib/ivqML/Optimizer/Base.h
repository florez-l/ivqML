// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Optimizer__Base__h__
#define __ivqML__Optimizer__Base__h__

#include <ivqML/Config.h>

namespace ivqML
{
  namespace Optimizer
  {
    /**
     */
    template< class _TModel >
    class Base
    {
    public:
      using TModel = _TModel;
      ivqML_TypeTraits( typename TModel::TReal );
      using Self = Base;

      using TDebugger =
        std::function<
          bool(
            const TNatural&, TModel*, const TReal&, const TReal&,
            const TReal*, const TReal*, const TNatural&,
            const TReal*, const TReal*, const TNatural&
            )
          >;

    protected:
      using TBatchRow = Eigen::Matrix< TNatural, 1, Eigen::Dynamic >;
      using TBatch    = Eigen::Map< TBatchRow >;
      using TBatches  = std::vector< TBatch >;
      using TShuffler = std::function< void( ) >;

    public:
      Base( );
      virtual ~Base( );

      TDebugger debugger( ) const;
      void set_debugger( TDebugger d );

      void set_data( TReal* X, TReal* Y, const TNatural& M );

      template< class _TBX, class _TBY >
      void set_data(
        Eigen::EigenBase< _TBX >& X, Eigen::EigenBase< _TBY >& Y
        );

      void fit( TModel& model );

    protected:
      virtual void _fit(
        TModel& model, TBatches& batches, TShuffler shuffler
        ) = 0;

    private:
      void _free( );

    protected:
      TReal*   m_X         { nullptr };
      TReal*   m_Y         { nullptr };
      TNatural m_M         { 0 };
      bool     m_OwnBuffer { true };

      ivqML_AttributeMacro( batch_size, BatchSize, TNatural, TNatural( 0 ) );
      ivqML_AttributeMacro( lambda1, Lambda1, TReal, TReal( 0 ) );
      ivqML_AttributeMacro( lambda2, Lambda2, TReal, TReal( 0 ) );

      TDebugger m_Debugger
        {
          [](
            const TNatural&, TModel*, const TReal&, const TReal&,
            const TReal*, const TReal*, const TNatural&,
            const TReal*, const TReal*, const TNatural&
            ) -> bool
          {
            return( false );
          }
        };
    };

  } // end namespace
} // end namespace

#include <ivqML/Optimizer/Base.hxx>

#endif // __ivqML__Optimizer__Base__h__

// eof - $RCSfile$
