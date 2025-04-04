// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__Optimizer__GradientDescent__h__
#define __ivqML__Optimizer__GradientDescent__h__

#include <ivqML/Config.h>

namespace ivqML
{
  namespace Optimizer
  {
    /**
     */
    template< class _TModel >
    class GradientDescent
    {
    public:
      using TModel = _TModel;
      using Self = GradientDescent;

      using TNatural = typename TModel::TNatural;
      using TReal    = typename TModel::TReal;
      using TMatrix  = typename TModel::TMatrix;
      using TMap     = Eigen::Map< const TMatrix >;

      using TBatch   = std::pair< TMap, TMap >;
      using TBatches = std::vector< TBatch >;

    public:
      GradientDescent(
        const TReal* Xtr, const TReal* Ytr,
        const TNatural& Mtr
        )
        {
          this->m_Xtr = Xtr;
          this->m_Ytr = Ytr;
          this->m_Xte = nullptr;
          this->m_Yte = nullptr;
          this->m_Mtr = Mtr;
          this->m_Mte = 0;
        }

      GradientDescent(
        const TReal* Xtr, const TReal* Ytr,
        const TReal* Xte, const TReal* Yte,
        const TNatural& Mtr, const TNatural& Mte
        )
        {
          this->m_Xtr = Xtr;
          this->m_Ytr = Ytr;
          this->m_Xte = Xte;
          this->m_Yte = Yte;
          this->m_Mtr = Mtr;
          this->m_Mte = Mte;
        }

      virtual ~GradientDescent( )
        {
        }

      void set_batch_size( const TNatural& s )
        {
          this->m_BatchSize = s;
        }

      void set_regularization( const TReal& l1, const TReal& l2 )
        {
        }

      void set_learning_rate( const TReal& a )
        {
          this->m_LearningRate = a;
        }

      void set_validation_to_normal( )
        {
        }

      void set_validation_to_leave_one_out( )
        {
        }

      void set_validation_to_kfold( const TNatural& k )
        {
        }

      void set_debugger( )
        {
        }

      void fit( TModel* model )
        {
          TNatural N = model->input_size( );
          TNatural O = model->output_size( );

          // Compute batches without copying input training data
          TBatches batches;
          TNatural batch_size = this->m_Mtr;
          if( 0 < this->m_BatchSize && this->m_BatchSize < this->m_Mtr )
            batch_size = this->m_BatchSize;
          TNatural n_batches = this->m_Mtr / batch_size;
          TNatural last_batch_size = this->m_Mtr % batch_size;
          const TReal* xtr = this->m_Xtr;
          const TReal* ytr = this->m_Ytr;
          for( TNatural b = 0; b < n_batches; ++b )
          {
            batches.push_back(
              TBatch(
                TMap( xtr, N, batch_size ), TMap( ytr, O, batch_size )
                )
              );
            xtr += batches.back( ).first.size( );
            ytr += batches.back( ).second.size( );
          } // end for
          if( last_batch_size > 0 )
            batches.push_back(
              TBatch(
                TMap( xtr, N, last_batch_size ), TMap( ytr, O, last_batch_size )
                )
              );

          this->_fit( model, batches );
        }

    protected:
      void _fit( TModel* model, const TBatches& batches )
        {
          TNatural S = model->size( );
          TMatrix G = TMatrix::Zero( 1, S );
          TMatrix sG = G;

          bool stop = false;
          TNatural t = 0;
          while( !stop )
          {
            t++;

            sG.fill( 0 );
            TReal Jtr = 0;
            for( const TBatch& batch: batches )
            {
              Jtr += model->gradient( G.data( ), batch.first, batch.second );
              sG += G;
              *model -= G * this->m_LearningRate;
            } // end for
            Jtr /= TReal( batches.size( ) );

            std::cout << t << " " << Jtr << " " << G << std::endl;
            stop = ( !( t < 10 ) );

          } // end while
        }

    protected:
      const TReal* m_Xtr { nullptr };
      const TReal* m_Ytr { nullptr };
      const TReal* m_Xte { nullptr };
      const TReal* m_Yte { nullptr };

      TNatural m_Mtr { 0 };
      TNatural m_Mte { 0 };

      TNatural m_BatchSize { 0 };
      TReal m_LearningRate { 1e-2 };
    };

  } // end namespace
} // end namespace

#endif // __ivqML__Optimizer__GradientDescent__h__

// eof - $RCSfile$
